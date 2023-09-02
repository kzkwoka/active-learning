import inspect
import numpy as np
import torch
import torch.nn as nn
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

def get_train_val(dataset: torch.utils.data.Dataset, valid_n: int | float=12.5) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Get indices for train and validations sets for repeating experiments

    Args:
        trainset (torch.utils.data.Dataset): training dataset
        valid_n (int | float, optional): percentage of data to be taken as validation set. Defaults to 12.5.

    Returns:
        tuple[List[int], List[int]]: list of training and validation indices
    """
    if valid_n > 1:
        valid_n = valid_n/100
    rand_sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
    batch_sampler = torch.utils.data.BatchSampler(rand_sampler, batch_size=int(valid_n*len(dataset)), drop_last=False)
    train_idx = []
    for i, batch in enumerate(iter(batch_sampler)):
        if i == 0:
            validation_idx = batch
        else:
            train_idx.extend(batch)
    return train_idx, validation_idx

def create_weighted_sampler(epoch_subset):
    targets = np.array(epoch_subset.dataset.targets)[np.hstack(epoch_subset.indices)]
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def train_epoch(model, device, optimizer, scheduler, loss_module, epoch_loader, val_loader):
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
    
    metrics = get_evaluation_metrics(model, device, val_loader, loss_module, running_loss, all_preds, all_targets)
    
    if scheduler:
        scheduler.step(metrics["validation_loss"])
    return metrics

def get_evaluation_metrics(model, device, loader, loss_module, training_loss, preds, targets):
    training_loss = training_loss/targets.shape[0]
    training_accuracy = multiclass_accuracy(preds, targets) # global
    training_f1_score = multiclass_f1_score(preds, targets) # global
    
    validation_loss, validation_accuracy, validation_f1_score = validate(model, device, loader, loss_module)
    
    log_str = f"Train acc: {training_accuracy:.3f} train loss: {training_loss:.4f} train f1: {training_f1_score:.4f} "\
                f"val acc: {validation_accuracy:.3f} val loss: {validation_loss:.4f} val f1: {validation_f1_score:.4f}"
    print(log_str)
    metric_names = ["training_loss", "training_accuracy", "training_f1_score", 
                        "validation_accuracy", "validation_loss", "validation_f1_score", 
                    ]
    metrics = {}
    for name in metric_names:
        metrics[name] = eval(name)
    return metrics

def validate(model, device, dataloader, loss_module, per_class=False, iter=0):
    try:
        running_loss = 0.0
        all_preds = torch.Tensor().to(device)
        all_targets = torch.Tensor().to(device)
        with torch.no_grad():
            for _, data in enumerate(dataloader):
                data, target = data[0].to(device), data[1].to(device)
                output = model(data)
                loss = loss_module(output, target)
                running_loss += loss.item()
                _, preds = torch.max(output.data, 1)
                all_preds = torch.cat((all_preds, preds), -1)
                all_targets = torch.cat((all_targets, target), -1)
            loss = running_loss/all_targets.shape[0]
            accuracy = multiclass_accuracy(all_preds, all_targets) # global
            f1_score = multiclass_f1_score(all_preds, all_targets) # global
            if not per_class:
                return loss, accuracy, f1_score
            else:
                accuracy_per_class = multiclass_accuracy(all_preds.to(torch.int64), all_targets.to(torch.int64), num_classes=len(dataloader.dataset.classes), average=None) # per class
                f1_score_per_class = multiclass_f1_score(all_preds.to(torch.int64), all_targets.to(torch.int64), num_classes=len(dataloader.dataset.classes), average=None) # per class
                f1_score_weighted = multiclass_f1_score(all_preds.to(torch.int64), all_targets.to(torch.int64), num_classes=len(dataloader.dataset.classes), average='weighted') # weighted sum
                return loss, accuracy, f1_score, accuracy_per_class, f1_score_per_class, f1_score_weighted
    except RuntimeError:
        if iter < 3:
            return validate(model, device, dataloader, loss_module, per_class, iter+1)

def filter_dict(func, kwarg_dict):
    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])
    common_args = sign.intersection(kwarg_dict.keys())
    filtered_dict = {key: kwarg_dict[key] for key in common_args}
    return filtered_dict

def generate_heuristic_sample(loader, n_samples, model, device, heuristic=None):
    if heuristic is None:
        return generate_random_sample(loader.dataset.imgs, n_samples)
    else:
        return eval(heuristic)(loader, n_samples, model, device)
    
def generate_random_sample(indices, n_samples):
    chosen = np.random.choice(indices, n_samples, replace=False)
    leftover = np.setdiff1d(indices, chosen, assume_unique=True)
    return chosen, leftover

def mc_dropout(loader, n_samples, model, device, iter=0):
    try:
        if len(loader.dataset) <= n_samples:
            return loader.dataset.imgs, []
        entropies = _get_mc_dropout_entropies(loader, model, device)
        largest = np.argpartition(entropies, -n_samples)[-n_samples:]
        chosen = [loader.dataset.imgs[i] for i in largest]
        leftover = np.setdiff1d(loader.dataset.imgs, chosen, assume_unique=True)
        return chosen, leftover
    except RuntimeError:
        if iter < 3:
            return mc_dropout(loader.dataset.imgs, n_samples, model, device, iter+1)

def _get_mc_dropout_entropies(loader, model, device):
    with torch.no_grad():
        model.eval()
        count = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.train(True)
                count += 1
        assert count > 0, 'We can only do models with dropout!'
        i = 0
        all_results = np.array([])
        for data in loader:
            data, _ = data[0].to(device), data[1].to(device)
            output = torch.Tensor().to(device)
            for _ in range(4):
                input = data.repeat(5, 1, 1, 1)
                preds = model(input).data
                output = torch.cat((output, preds), 0)
            average_output = output.view(20, data.size(0), -1).mean(dim=0) 
            probs = torch.softmax(average_output,axis=1)
            entropy = (-probs * probs.log()).sum(dim=1, keepdim=True)
            all_results = np.append(all_results, entropy.cpu().numpy())
            i+=1
        return all_results

def largest_margin(loader, n_samples, model, device, iter=0):
    try:
        if len(loader.dataset) <= n_samples:
            return loader.imgs, []
        with torch.no_grad():
            diff = np.array([])
            for data in loader:
                data, _ = data[0].to(device), data[1].to(device)
                output = model(data)
                probs = torch.softmax(output, axis=1)
                batch_diff = torch.max(probs.data, 1)[0] - torch.min(probs.data, 1)[0]
                diff = np.append(diff, batch_diff.cpu().numpy())
            smallest = np.argpartition(diff, n_samples)[:n_samples]
            chosen = [loader.dataset.imgs[i] for i in smallest]
            leftover = np.setdiff1d(loader.dataset.imgs, chosen, assume_unique=True)
            return chosen, leftover
    except RuntimeError:
        if iter < 3:
            return largest_margin(loader.dataset.imgs, n_samples, model, device, iter+1)