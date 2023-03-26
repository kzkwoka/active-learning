import torch
import pandas as pd
import numpy as np
from heuristics import generate_heuristic_sample
from loading_params import use_base_dicts
from validating import validate


def train(model, device, optimizer, scheduler, loss_module, sub_epochs, epoch_loader, val_loader):
    for sub_epoch in range(sub_epochs):
        model.train()  # turn on training mode
        total = 0   # total n of samples seen
        correct = 0   # total n of coreectly classified samples
        running_loss = 0.0
        for _, data in enumerate(epoch_loader):
            data, target = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()  # zero out the gradients
            output = model(data)  # get the output of the net
            loss = loss_module(output, target)  # calculate the loss 
            running_loss += loss.item()  # add the loss on the subset batch
            _, preds = torch.max(output.data, 1)  # get the predictions
            correct += (preds == target).sum().item() # add the n of correctly classified samples
            total += target.size(0) # add the total of samples seen
            loss.backward()  # backpropagate the weights
            optimizer.step()  # optimize
            del data, target, output, preds
        t_loss = running_loss/total
        t_acc = 100. * correct/total
        val_loss, val_acc = validate(model, device, val_loader, loss_module)
        print(f"Sub epoch {sub_epoch} train acc: {t_acc:.2f} train loss: {t_loss:.4f} val acc: {val_acc:.2f} val loss: {val_loss:.4f}")
        scheduler.step(val_loss)
    return t_loss, t_acc, val_loss, val_acc

def base_learn(model, device, optimizer, scheduler, loss_module,
               train_data, train_idx, val_idx, test_loader, 
               initial_dict, optim_dict, sched_dict, sub_epochs = 20):
    model, optimizer, scheduler = use_base_dicts(model, optimizer, scheduler, initial_dict, optim_dict, sched_dict)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data, val_idx),
        batch_size=64,
        num_workers=8)
    validation_loss, validation_acc = [], []
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    batch_idx = train_idx
    print(f"Training on {len(batch_idx)} samples")
    
    epoch_subset =  torch.utils.data.Subset(train_data, batch_idx) # get epoch subset 
    epoch_loader =  torch.utils.data.DataLoader(epoch_subset, batch_size=64, num_workers=8, shuffle=True) # convert into loader
    # consider retraining multiple times
    t_loss, t_acc, val_loss, val_acc = train(model, device, optimizer, scheduler, loss_module, sub_epochs, epoch_loader, val_loader)

    train_loss.append(t_loss)
    train_acc.append(t_acc)

    validation_loss.append(val_loss)
    validation_acc.append(val_acc)
    
    ts_loss, ts_acc = validate(model, device, test_loader, loss_module)
    test_loss.append(ts_loss)
    test_acc.append(ts_acc)

    print(f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f} ",
      f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}")
    return train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc

def run_learning(net, device, optimizer, scheduler, loss_module,
                 trainset, train_idx_df, val_idx_df, testset, 
                 initial_dict, optim_dict, sched_dict, epochs=5, sub_epochs=20):

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        num_workers=8)
    result = []
    
    for i in range(epochs):
        i = str(i)
        train_loss, train_acc, validation_loss, validation_acc, test_loss, test_acc = \
            base_learn(net, device, optimizer, scheduler, loss_module, trainset,
                       train_idx_df[i].to_list(), val_idx_df[i].to_list(), test_loader,
                       initial_dict, optim_dict, sched_dict, sub_epochs)
        result.extend(make_rows("train", i, train_loss, train_acc))
        result.extend(make_rows("val", i, validation_loss, validation_acc))
        result.extend(make_rows("test", i, test_loss, test_acc))
    
    columns = ["dset", "epoch", "subepoch", "loss", "acc"]
    df = pd.DataFrame(data=result, columns=columns)
    return df

def make_rows(dset, epoch, loss, acc):
    temp = []
    for idx, (loss, acc) in enumerate(zip(loss, acc)):
            row = [dset, epoch, idx, loss, acc]
            temp.append(row)
    return temp
    
def active_learn(model, device, optimizer, scheduler, loss_module,
                 train_data, train_idx, val_idx, test_loader,
                 initial_train_idx, experiment_id,
                 initial_dict, optim_dict, sched_dict, sub_epochs = 20,
                 heuristic=None, n_batches=20):
    #TODO: configure active learning
    model, optimizer, scheduler = use_base_dicts(model, optimizer, scheduler, initial_dict, optim_dict, sched_dict)
    batch_len = int((1./n_batches)*len(train_idx)) # calculate length of 5% of training data
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_data, val_idx),
        batch_size=64,
        num_workers=8)
    validation_loss, validation_acc = [], []
    train_loss, train_acc = [], []
    for epoch in range(n_batches):
        print("Epoch: ", epoch)
        if epoch == 0:
        # if False:
            # batch_idx, train_idx = generate_random_sample(train_idx, batch_len) # equivalent to first (labeled) dataset
            batch_idx = initial_train_idx
        else:
            for group in optimizer.param_groups:
                group['lr'] = 0.001
            old_idx = batch_idx 
            batch_idx, train_idx = generate_heuristic_sample(train_idx, batch_len, model, heuristic, train_data) # equivalent to asking for labelling
            # consider joining the samples
            batch_idx = np.append(old_idx,batch_idx)
        print(f"Training on {len(batch_idx)} samples")
        
        epoch_subset =  torch.utils.data.Subset(train_data, batch_idx) # get epoch subset 
        epoch_loader =  torch.utils.data.DataLoader(epoch_subset, batch_size=64, num_workers=8, shuffle=True) # convert into loader
        # consider retraining multiple times
        t_loss, t_acc, val_loss, val_acc = train(model, device, optimizer, scheduler, loss_module, sub_epochs, epoch_loader, val_loader)
            
        train_loss.append(t_loss)
        train_acc.append(t_acc)

        # val_loss, val_acc = validate(model, val_loader)
        validation_loss.append(val_loss)
        validation_acc.append(val_acc)

        print(f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f} ",
          f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}")
        if heuristic is not None:
            torch.save(model.state_dict(), f'{experiment_id}/epoch{epoch}_{heuristic.__name__}.pt')
        else:
            torch.save(model.state_dict(), f'{experiment_id}/epoch{epoch}_random.pt')
    return train_loss, train_acc, validation_loss, validation_acc