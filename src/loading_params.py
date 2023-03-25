import torch

def get_train_val(trainset):
    rand_sampler = torch.utils.data.RandomSampler(trainset, replacement=False)
    batch_sampler = torch.utils.data.BatchSampler(rand_sampler,
                                                       batch_size=int(0.1*len(trainset)),
                                                       drop_last=False)
    train_idx = []
    for i, batch in enumerate(iter(batch_sampler)):
        if i == 0:
            validation_idx = batch
        else:
            train_idx.extend(batch)
    return train_idx, validation_idx