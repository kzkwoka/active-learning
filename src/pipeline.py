import json
import logging as log
from cuda_setup import load_device, set_seeds
from image_datasets import load_dataset
from loading_models import load_model, load_modules
from loading_params import load_base_dicts, load_indices
from training import run_learning

def run(model, dataset, epochs=5, sub_epochs=30, is_active_learning=False, heuristic=None, n_batches=20, resume_from=0):
    param_path = "params/"
    log.info("Training started")
    device = load_device()
    set_seeds()
    
    trainset, testset = load_dataset(dataset, model)
    train_idx_df, val_idx_df, labeled_idx_df, unlabeled_idx_df = load_indices(path=param_path, dataset=trainset)
    
    net = load_model(model, device, len(trainset.classes))
    optimizer, scheduler = load_modules(net.parameters())
    initial_dict, optim_dict, sched_dict = load_base_dicts(net, optimizer, scheduler, param_path)
    
    if is_active_learning:
        train_idx_df = unlabeled_idx_df
    best_metrics = run_learning(net, device, optimizer, scheduler, 
                      trainset, train_idx_df, val_idx_df, testset, 
                      initial_dict, optim_dict, sched_dict, 
                      epochs=epochs, sub_epochs=sub_epochs, active_learning=is_active_learning, 
                      heuristic=heuristic, labeled_idx_df=labeled_idx_df, n_batches=n_batches, resume_from=resume_from)
    
    return best_metrics

#TODO: run all experiments

if __name__ == "__main__":
    models = ["EFFNETV2S","VGG16", "OWN"]
    datasets = ["CIFAR10", "FashionMNIST", "FastFoodV2", "PlantVillage"]
    heuristics = ["largest_margin", "smallest_margin", "least_confidence", "mc_dropout", "representative_sampling", "representative_mc_dropout",None]
    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'

    resume_from = 0
    model = models[0]
    dataset = datasets[-1]
    experiments = 5
    epochs = 15
    active = True
    heuristic = heuristics[0] # if active 
    n_batches = 10 # if active
    if active:
        log.basicConfig(filename=f'./logs/{model}_{dataset}_{heuristic}.log', filemode='a+', format=FORMAT, level=log.DEBUG)
        best_metrics = run(model=model, dataset=dataset, epochs=experiments, sub_epochs=epochs, is_active_learning=active, heuristic=heuristic, n_batches=n_batches, resume_from=resume_from) 
        # with open(f"results/{model}_{dataset}_{heuristic}.json", "w") as f:
        #     json.dump(best_metrics, f)
    
    else:
        log.basicConfig(filename=f'./logs/{model}_{dataset}.log', filemode='a+', format=FORMAT, level=log.DEBUG)
        best_metrics = run(model=model, dataset=dataset, epochs=experiments, sub_epochs=epochs, is_active_learning=active, resume_from=resume_from) 
        # with open(f"results/{model}_{dataset}.json", "w") as f:
        #     json.dump(best_metrics, f)
    



