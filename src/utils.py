import numpy as np
import torch


def create_weighted_sampler(epoch_subset):
    targets = epoch_subset.dataset.targets[np.hstack(epoch_subset.indices)]
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

def make_rows(dset, epoch, loss, acc):
    temp = []
    for idx, (loss, acc) in enumerate(zip(loss, acc)):
            row = [dset, epoch, idx, loss, acc]
            temp.append(row)
    return temp