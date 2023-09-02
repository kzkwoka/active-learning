import os
import torch
import pandas as pd
import numpy as np
import logging as log
from heuristics import generate_heuristic_sample
from loading_params import use_base_dicts
from utils import create_weighted_sampler
from validating import get_evaluation_metrics

from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

def run_learning(net, device, optimizer, scheduler, 
                 trainset, train_idx_df, val_idx_df, testset, 
                 initial_dict, optim_dict, sched_dict, epochs=5, sub_epochs=20, active_learning=False, heuristic=None, labeled_idx_df=None, n_batches=20, resume_from=0):

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=TEST_BATCH_SIZE,
        num_workers=4)
    best_metrics = []
    
    for i in range(epochs):
        if i >= resume_from:
            i = str(i)
            log.info(f"Starting experiment {i}")
            if active_learning: 
                metrics = active_learn(net, device, optimizer, scheduler, trainset,
                                    train_idx_df[i].to_list(), val_idx_df[i].to_list(), test_loader, 
                                    labeled_idx_df[i], initial_dict, optim_dict, sched_dict, sub_epochs, i,
                                    heuristic, n_batches)
            else:
                metrics = base_learn(net, device, optimizer, scheduler, trainset,
                        train_idx_df[i].to_list(), val_idx_df[i].to_list(), test_loader,
                        initial_dict, optim_dict, sched_dict, sub_epochs, i)
                best_metrics.append(metrics)
    
    return best_metrics

def base_learn(model, device, optimizer, scheduler, 
               train_data, train_idx, val_idx, test_loader, 
               initial_dict, optim_dict, sched_dict, sub_epochs = 20, experiment_id=0):
    
    model, optimizer, scheduler = use_base_dicts(model, optimizer, scheduler, initial_dict, optim_dict, sched_dict)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data, val_idx),
        batch_size=VALID_BATCH_SIZE,
        num_workers=4)
    
    loss_module = torch.nn.CrossEntropyLoss()
    log.info(f"Loaded Cross Entropy Loss")
    
    batch_idx = train_idx
    log.info(f"Training on {len(batch_idx)} samples")
    
    epoch_subset =  torch.utils.data.Subset(train_data, batch_idx) # get epoch subset 
    sampler = create_weighted_sampler(epoch_subset)
    epoch_loader =  torch.utils.data.DataLoader(epoch_subset, batch_size=TRAIN_BATCH_SIZE, 
                                                num_workers=4, sampler=sampler) # convert into loader
    
    metrics = train(model, device, optimizer, scheduler, loss_module, sub_epochs, 
                                                              epoch_loader, val_loader, test_loader, experiment_id)

    return metrics

def train(model, device, optimizer, scheduler, loss_module, sub_epochs, 
          epoch_loader, val_loader, test_loader, experiment_id):
    best_metrics = {"validation_accuracy": 0}
    
    for sub_epoch in range(sub_epochs):
        metrics = _train_epoch(model, device, optimizer, scheduler, loss_module, sub_epoch, epoch_loader, val_loader, test_loader)
        
        if metrics["validation_accuracy"] > best_metrics["validation_accuracy"]:
            log.info(f"Saving best model with val_acc: {metrics['validation_accuracy']}")
            torch.save(model.state_dict(), f'logs/{experiment_id}_best_base.pt')
            best_metrics = metrics
        
    return metrics

def train_active(model, device, optimizer, scheduler, loss_module, sub_epochs, 
          epoch_loader, val_loader, test_loader, experiment_id, heuristic  ):
    best_metrics = {"validation_accuracy": 0}
    
    for epoch in range(sub_epochs):
        metrics = _train_epoch(model, device, optimizer, scheduler, loss_module, epoch, epoch_loader, val_loader, test_loader)
        
        if metrics["validation_accuracy"] > best_metrics["validation_accuracy"]:
            log.info(f"Saving best model with val_acc: {metrics['validation_accuracy']}")
            
            # if not os.path.exists( f'logs/{experiment_id}'):
            #     os.makedirs( f'logs/{experiment_id}')

            # if heuristic is not None:
            #     torch.save(model.state_dict(), f'logs/{experiment_id}/epoch{epoch}_{heuristic}.pt')
            # else:
            #     torch.save(model.state_dict(), f'logs/{experiment_id}/epoch{epoch}_random.pt')
                
            best_metrics = metrics
    return metrics

def _train_epoch(model, device, optimizer, scheduler, loss_module, sub_epoch, epoch_loader, val_loader, test_loader):
    model.train()  # turn on training mode
    running_loss = 0.0
    all_preds = torch.Tensor().to(device)
    all_targets = torch.Tensor().to(device)
    for _, data in enumerate(epoch_loader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()  # zero out the gradients
        output = model(data)  # get the output of the net
        loss = loss_module(output, target)  # calculate the loss 
        running_loss += loss.item()  # add the loss on the subset batch
        _, preds = torch.max(output.data, 1)  # get the predictions
        all_preds = torch.cat((all_preds, preds), -1)
        all_targets = torch.cat((all_targets, target), -1)
        loss.backward()  # backpropagate the weights
        optimizer.step()  # optimize
    model.eval()
    metrics = get_evaluation_metrics(model, device, val_loader, test_loader, loss_module, sub_epoch,
                        running_loss, all_preds, all_targets)
                
    scheduler.step(metrics["validation_loss"])
    return metrics

def active_learn(model, device, optimizer, scheduler,
                 train_data, train_idx, val_idx, test_loader,
                 initial_train_idx, 
                 initial_dict, optim_dict, sched_dict, 
                 sub_epochs=20, experiment_id=0,
                 heuristic=None, n_batches=20):
    
    model, optimizer, scheduler = use_base_dicts(model, optimizer, scheduler, initial_dict, optim_dict, sched_dict)
    batch_len = int((1./n_batches)*len(train_idx)) # calculate length of 100/n_batches % of training data
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data, val_idx),
        batch_size=VALID_BATCH_SIZE,
        num_workers=4)
    
    loss_module = torch.nn.CrossEntropyLoss()
    log.info(f"Loaded Cross Entropy Loss")
    
    for epoch in range(n_batches):
        #TODO: consider resetting the nn here 
        log.info(f"Epoch: {epoch}")
        if epoch == 0:
            batch_idx = initial_train_idx # equivalent to first (labeled) dataset
        else:
            for group in optimizer.param_groups:
                group['lr'] = 0.001 #reset learning rate
            old_idx = batch_idx 
            batch_idx, train_idx = generate_heuristic_sample(train_idx, batch_len, model, device, heuristic, train_data, batch_idx) # equivalent to asking for labeling
            batch_idx = np.append(old_idx,batch_idx)

        log.info(f"Training on {len(batch_idx)} samples") 
        
        epoch_subset =  torch.utils.data.Subset(train_data, batch_idx) # get epoch subset 
        sampler = create_weighted_sampler(epoch_subset)
        epoch_loader =  torch.utils.data.DataLoader(epoch_subset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, sampler=sampler) # convert into loader

        metrics = train_active(model, device, optimizer, scheduler, loss_module, sub_epochs, 
                                                              epoch_loader, val_loader, test_loader, experiment_id, heuristic)
            
        
        
    return metrics
