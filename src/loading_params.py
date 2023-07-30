import copy
import os
from pathlib import Path
from typing import List
import pandas as pd
import torch
import logging as log

from heuristics import generate_random_sample

def get_train_val(trainset: torch.utils.data.Dataset, valid_n: int | float=12.5) -> tuple[List[int], List[int]]:
    """Get indices for train and validations sets for repeating experiments

    Args:
        trainset (torch.utils.data.Dataset): training dataset
        valid_n (int | float, optional): percentage of data to be taken as validation set. Defaults to 12.5.

    Returns:
        tuple[List[int], List[int]]: list of training and validation indices
    """     
    valid_n = valid_n/100
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
        n (int, optional): number of times for the generation to be performed. Defaults to 5.
        path (str | Path, optional): path for the resulting files to be saved at. Defaults to "".
    """    
    if dataset is None:
        raise 
    train_idx_df = pd.DataFrame()
    val_idx_df = pd.DataFrame()
    labeled_idx_df = pd.DataFrame()
    unlabeled_idx_df = pd.DataFrame()
    for i in range(n):
        train_idx, validation_idx = get_train_val(dataset, valid_n=valid_n)
        train_idx_df[i] = train_idx
        val_idx_df[i] = validation_idx
        labeled_idx, unlabeled_idx = generate_random_sample(train_idx, int(len(train_idx)*valid_n/100))
        labeled_idx_df[i] = labeled_idx
        unlabeled_idx_df[i] = unlabeled_idx

    log.info(f"Generated {n} splits - {valid_n}% validation set - {valid_n}% labeled set")
    
    train_idx_df.to_csv(f"{path}train_idx.csv", index=False)
    val_idx_df.to_csv(f"{path}val_idx.csv", index=False)
    labeled_idx_df.to_csv(f"{path}labeled_idx.csv", index=False)
    unlabeled_idx_df.to_csv(f"{path}unlabeled_idx.csv", index=False)
    
def load_indices(path: str | Path="", n : int = 5, dataset: torch.utils.data.Dataset | None = None,
                 valid_n: int | float=12.5) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the generated training and validation indices

    Args:
        path (str | Path, optional): path where resulting files are stored. Defaults to "".
        n (int, optional): n fold generation of train-validation split. Defaults to 5.
        dataset (torch.utils.data.Dataset | None, optional): dataset from which the folds will be created. Defaults to None.
        valid_n (int | float): percentage of training data to be placed in validation. Defaults to 12.5.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: dataframes with the training and validation indices
    """    
    exist = os.path.exists(f"{path}train_idx.csv") and os.path.exists(f"{path}train_idx.csv") and os.path.exists(f"{path}labeled_idx.csv") and os.path.exists(f"{path}unlabeled_idx.csv") 
    if not exist:
        generate_indices(path=path, n=n, dataset=dataset, valid_n=valid_n)

    train_idx_df = pd.read_csv(f"{path}train_idx.csv")
    log.info(f"Training indices read from {path}train_idx.csv")
    
    val_idx_df = pd.read_csv(f"{path}val_idx.csv")
    log.info(f"Validation indices read from {path}val_idx.csv")
    
    labeled_idx_df = pd.read_csv(f"{path}labeled_idx.csv")
    log.info(f"Initially labeled indices read from {path}val_idx.csv")
    
    unlabeled_idx_df = pd.read_csv(f"{path}unlabeled_idx.csv")
    log.info(f"Initially unlabeled indices read from {path}val_idx.csv")
    return train_idx_df, val_idx_df, labeled_idx_df, unlabeled_idx_df

def generate_base_dicts(model, optimizer, scheduler, path: str | Path=""):
    initial_dict = copy.deepcopy(model.state_dict())
    optim_dict = copy.deepcopy(optimizer.state_dict())
    sched_dict = copy.deepcopy(scheduler.state_dict())

    torch.save(initial_dict, f"{path}model_dict.pt")
    torch.save(optim_dict, f"{path}optimizer_dict.pt")
    torch.save(sched_dict, f"{path}scheduler_dict.pt")
    
    log.info("Generated supporting modules base weights")

def load_base_dicts(model, optimizer, scheduler, path: str | Path=""):
    exist = (os.path.exists(f"{path}model_dict.pt") and 
             os.path.exists(f"{path}optimizer_dict.pt") and 
             os.path.exists(f"{path}scheduler_dict.pt"))
    if not exist:
        generate_base_dicts(model, optimizer, scheduler, path)
    initial_dict = torch.load(f"{path}model_dict.pt")
    optim_dict = torch.load(f"{path}optimizer_dict.pt")
    sched_dict = torch.load(f"{path}scheduler_dict.pt")
    log.info("Read supporting modules base weights")
    return initial_dict, optim_dict, sched_dict

def use_base_dicts(model, optimizer, scheduler, initial_dict, optim_dict, sched_dict):
    model.load_state_dict(initial_dict)
    optimizer.load_state_dict(optim_dict)
    scheduler.load_state_dict(sched_dict)
    log.info("Used supporting modules base weights")
    return model, optimizer, scheduler

