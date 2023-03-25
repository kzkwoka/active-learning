import copy
import os
from pathlib import Path
from typing import List
import pandas as pd
import torch

def get_train_val(trainset: torch.utils.data.Dataset, valid_n: int | float=12.5) -> tuple[List[int], List[int]]:
    """Get indices for train and validations sets for repeating experiments

    Args:
        trainset (torch.utils.data.Dataset): training dataset
        valid_n (int | float, optional): percentage of data to be taken as validation set. Defaults to 12.5.

    Returns:
        tuple[List[int], List[int]]: list of training and validation indices
    """     
    valid_n = valid_n/10
    rand_sampler = torch.utils.data.RandomSampler(trainset, replacement=False)
    batch_sampler = torch.utils.data.BatchSampler(rand_sampler,
                                                       batch_size=int(valid_n*len(trainset)),
                                                       drop_last=False)
    train_idx = []
    for i, batch in enumerate(iter(batch_sampler)):
        if i == 0:
            validation_idx = batch
        else:
            train_idx.extend(batch)
    return train_idx, validation_idx

def generate_indices(n : int = 5, path : str | Path="", dataset: torch.utils.data.Dataset | None = None,
                     valid_n: int | float=12.5):
    """Generate and save the dataframes with training and validation indices n times.

    Args:
        n (int, optional): number of times for the generatiion to be performed. Defaults to 5.
        path (str | Path, optional): path for the resulting files to be saved at. Defaults to "".
    """    
    if dataset is None:
        raise 
    train_idx_df = pd.DataFrame()
    val_idx_df = pd.DataFrame()
    for i in range(n):
        train_idx, validation_idx = get_train_val(dataset, valid_n=valid_n)
        train_idx_df[i] = train_idx
        val_idx_df[i] = validation_idx
    train_idx_df.to_csv(f"{path}train_idx.csv", index=False)
    val_idx_df.to_csv(f"{path}val_idx.csv", index=False)
    
def load_indices(path: str | Path="", n : int = 5, dataset: torch.utils.data.Dataset | None = None,
                 valid_n: int | float=12.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """load the generated training and validation indices

    Args:
        path (str | Path, optional): path where resulting files are stored. Defaults to "".

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: dataframes with the training and validation indices
    """    
    exist = os.path.exists(f"{path}train_idx.csv") and os.path.exists(f"{path}train_idx.csv")
    if not exist:
        generate_indices(path=path, n=n, dataset=dataset, valid_n=valid_n)
    train_idx_df = pd.read_csv(f"{path}train_idx.csv")
    val_idx_df = pd.read_csv(f"{path}val_idx.csv")
    return train_idx_df, val_idx_df

def generate_base_dicts(model, optimizer, scheduler, initial_dict, optim_dict, sched_dict):
    initial_dict = copy.deepcopy(model.state_dict())
    optim_dict = copy.deepcopy(optimizer.state_dict())
    sched_dict = copy.deepcopy(scheduler.state_dict())

    torch.save(initial_dict, "model_dict.pt")
    torch.save(optim_dict, "optimizer_dict.pt")
    torch.save(sched_dict, "scheduler_dict.pt")

def load_base_dicts():
    initial_dict =torch.load("model_dict.pt")
    optim_dict = torch.load("optimizer_dict.pt")
    sched_dict = torch.load("scheduler_dict.pt")
    return initial_dict, optim_dict, sched_dict

def use_base_dicts(model, optimizer, scheduler, initial_dict, optim_dict, sched_dict):
    model.load_state_dict(initial_dict)
    optimizer.load_state_dict(optim_dict)
    scheduler.load_state_dict(sched_dict)
    return model, optimizer, scheduler

#TODO: add path variable and check if files exist like for indices