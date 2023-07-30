import time
import torch
import numpy as np
import torch.nn as nn

def generate_heuristic_sample(indices, n_samples, model, device, heuristic=None, data=None, labeled_indices=None):
    if heuristic is None:
        return generate_random_sample(indices, n_samples)
    else:
        return eval(heuristic)(indices, n_samples, model, data, device, labeled_indices)

def generate_random_sample(indices, n_samples):
    chosen = np.random.choice(indices, n_samples, replace=False)
    leftover = np.setdiff1d(indices, chosen, assume_unique=True)
    return chosen, leftover

def mc_dropout(indices, n_samples, model, data, device, labeled_indices=None):
    if len(indices) <= n_samples:
        return indices, []
    entropies = _get_mc_dropout_entropies(data, indices, model, device)
    largest = np.argpartition(entropies, -n_samples)[-n_samples:]
    chosen = [indices[i] for i in largest]
    leftover = np.setdiff1d(indices, chosen, assume_unique=True)
    return chosen, leftover

def _get_mc_dropout_entropies(data, indices, model, device):
    with torch.no_grad():
        heuristic_loader = torch.utils.data.DataLoader(
          torch.utils.data.Subset(data, indices),
          batch_size=64,
          num_workers=8
        )
        model.eval()
        count = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.train(True)
                count += 1
        assert count > 0, 'We can only do models with dropout!'
        i = 0
        all_results = np.array([])
        for data in heuristic_loader:
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

def representative_sampling(indices, n_samples, model, data, device,labeled_indices=None):
    distances = _get_representative_sampling_distances(data, indices, labeled_indices, device)
    largest = np.argpartition(distances, -n_samples)[-n_samples:]
    chosen = [indices[i] for i in largest]
    leftover = np.setdiff1d(indices, chosen, assume_unique=True)
    return chosen, leftover

def _calculate_centroid(loader, device):
    unlabeled_sum = None
    count = 0
    for data in loader:
        data = data[0].to(device).view(data[0].shape[0], -1)
        if unlabeled_sum is None:
            unlabeled_sum = torch.sum(data, 0)
        else:
            unlabeled_sum += torch.sum(data, 0)
        count += data.shape[0]
    return unlabeled_sum/count
    
def _get_representative_sampling_distances(data, non_labeled_idx, labeled_idx, device):
    unlabeled_loader = torch.utils.data.DataLoader(
          torch.utils.data.Subset(data, non_labeled_idx),
          batch_size=64,
          num_workers=8
        )
    labeled_loader = torch.utils.data.DataLoader(
          torch.utils.data.Subset(data, labeled_idx),
          batch_size=64,
          num_workers=8
        )
    unlabeled_centroid = _calculate_centroid(unlabeled_loader, device)
    labeled_centroid = _calculate_centroid(labeled_loader, device)
    
    all_r = np.array([])
    for data in unlabeled_loader:
        data = data[0].to(device).view(data[0].shape[0], -1)
        r = torch.cdist(data, torch.cat((unlabeled_centroid.view(1,-1),labeled_centroid.view(1,-1)))).diff()
        all_r = np.append(all_r, r.cpu().numpy())
    
    return all_r

def representative_mc_dropout(indices, n_samples, model, data, device,labeled_indices=None):
    distances = _get_representative_sampling_distances(data, indices, labeled_indices, device)
    entropies = _get_mc_dropout_entropies(data, indices, model, device)
    dist_rank = np.argsort(distances).argsort()
    ent_rank = np.argsort(entropies).argsort()
    combined_rank = dist_rank + ent_rank
    largest = np.argsort(combined_rank)[-n_samples:]
    chosen = [indices[i] for i in largest]
    leftover = np.setdiff1d(indices, chosen, assume_unique=True)
    return chosen, leftover
    
def largest_margin(indices, n_samples, model, data, device,labeled_indices=None):
    if len(indices) <= n_samples:
        return indices, []
    with torch.no_grad():
        heuristic_loader = torch.utils.data.DataLoader(
          torch.utils.data.Subset(data, indices),
          batch_size=64,
          num_workers=8
        )
        diff = np.array([])
        for data in heuristic_loader:
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            probs = torch.softmax(output, axis=1)
            batch_diff = torch.max(probs.data, 1)[0] - torch.min(probs.data, 1)[0]
            diff = np.append(diff, batch_diff.cpu().numpy())
        #choose n_samples with smallest ?
        
        smallest = np.argpartition(diff, n_samples)[:n_samples]
        chosen = indices[smallest]
        leftover = np.setdiff1d(indices, chosen, assume_unique=True)
        return chosen, leftover
    
def smallest_margin(indices, n_samples, model, data, device,labeled_indices=None):
    if len(indices) <= n_samples:
        return indices, []
    with torch.no_grad():
        heuristic_loader = torch.utils.data.DataLoader(
          torch.utils.data.Subset(data, indices),
          batch_size=64,
          num_workers=8
        )
        diff = np.array([])
        for data in heuristic_loader:
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            probs = torch.softmax(output, axis=1)
            top2 = torch.topk(probs.data, 2).values
            batch_diff = top2[:,0] - top2[:,1]
            diff = np.append(diff, batch_diff.cpu().numpy())
        #choose n_samples with smallest ?
        smallest = np.argpartition(diff, n_samples)[:n_samples]
        chosen = indices[smallest]
        leftover = np.setdiff1d(indices, chosen, assume_unique=True)
        return chosen, leftover
    
def least_confidence(indices, n_samples, model, data, device,labeled_indices=None):
    if len(indices) <= n_samples:
        return indices, []
    with torch.no_grad():
        heuristic_loader = torch.utils.data.DataLoader(
          torch.utils.data.Subset(data, indices),
          batch_size=64,
          num_workers=8
        )
        max_probs = np.array([])
        for data in heuristic_loader:
            data, target = data[0].to(device), data[1].to(device)
            output = model(data)
            probs = torch.softmax(output, axis=1)
            max_prob = torch.max(output.data, 1)[0]
            max_probs = np.append(max_probs, max_prob.cpu().numpy())
        #choose n_samples with smallest ?
        smallest = np.argpartition(max_probs, n_samples)[:n_samples]
        chosen = indices[smallest]
        leftover = np.setdiff1d(indices, chosen, assume_unique=True)
        return chosen, leftover

def score_svm_heuristic(indices, n_samples, model, data, device):
    # if len(indices) <= n_samples:
    #     return indices, []
    # # global svm, train_svm
    # svm_results = np.array([])
    # # if train_svm:
    # if True:
    #     heuristic_loader = torch.utils.data.DataLoader(
    #           torch.utils.data.Subset(data, indices),
    #           batch_size=64,
    #           num_workers=8
    #         )
    #     for data in heuristic_loader:
    #         data, target = data[0].to(device), data[1].to(device)
    #         output = model(data)
    #         svm_result = svm(output)
    #         svm_results = np.append(svm_results, svm_result.detach().cpu().numpy())
    #     #choose n_samples with smallest ?
    #     smallest = np.argpartition(svm_results, n_samples)[:n_samples]
    #     chosen = indices[smallest]
    #     leftover = np.setdiff1d(indices, chosen, assume_unique=True)
    #     # train_svm = False
    #     return chosen, leftover
    # else:
    #     with torch.no_grad():
    #         heuristic_loader = torch.utils.data.DataLoader(
    #           torch.utils.data.Subset(data, indices),
    #           batch_size=64,
    #           num_workers=8
    #         )
    #         for data in heuristic_loader:
    #             data, target = data[0].to(device), data[1].to(device)
    #             output = model(data)
    #             svm_result = svm(output)
    #             svm_results = np.append(svm_results, svm_result.cpu().numpy())
    #         #choose n_samples with largest ?
    #         smallest = np.argpartition(svm_results, n_samples)[-n_samples:]
    #         chosen = indices[smallest]
    #         leftover = np.setdiff1d(indices, chosen, assume_unique=True)
    #         return chosen, leftover
    pass

def knn_svm_heuristic():
    # class KNNModel(nn.Module):
    # """
    #     This is our Nearest Neighbour "neural network".
    # """

    # def __init__(self, base_data, k=1):
    #     super(KNNModel, self).__init__()
    #     self.base_data = base_data.half().cuda()         # We probably have to rewrite this part of the code 
    #                                                      # as larger datasets may not entirely fit into the GPU memory. maybe downsampling?
    #     n_data = self.base_data.size(0)
    #     self.base_data = self.base_data.view(n_data, -1) # Flatten the train data.
    #     self.base_data_norm = (self.base_data*self.base_data).sum(dim=1)
    #     self.K = k
    #     self.norm = 2
    
    # def forward(self, x, **kwargs):
    #     n_samples = x.size(0)
    #     x = x.data.view(n_samples, -1).half() # flatten to vectors.
    #     base_data = self.base_data
    #     base_norm = self.base_data_norm
    #     ref_size = base_data.size(0)

    #     x_norm = (x*x).sum(dim=1)
    #     diffs = base_norm.unsqueeze(0).expand(n_samples, ref_size) + x_norm.unsqueeze(1).expand(n_samples, ref_size) - 2*x.matmul(base_data.t())
    #     diffs.sqrt_().detach_()

    #     output, _ = torch.topk(diffs, self.K, dim=1, largest=False, sorted=True)

    #     return output.float()
    
    # def preferred_name(self):
    #     return '%d-NN'%self.K

    # def output_size(self):
    #     return torch.LongTensor([1, self.K])
    pass


# if __name__ == "__main__":
#     from cuda_setup import load_device
#     from image_datasets import load_cifar
    
#     train, test = load_cifar("VGG16")
#     n_samples = 10
#     device = load_device()
#     data = torch.from_numpy(train.data).to(device).type(torch.float32)
#     # data = torch.from_numpy(test.data).to(device)
    
# #     # KMeans(data, K=n_samples)
# #     _, cluster_centers = kmeans(X=data, num_clusters=64, distance='euclidean', device=device)
