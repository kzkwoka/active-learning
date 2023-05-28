import logging as log
from cuda_setup import load_device, set_seeds
from image_datasets import load_dataset
from loading_models import load_model, load_modules
from loading_params import load_base_dicts, load_indices
from training import run_learning

def run(model, dataset, epochs=5, sub_epochs=30, is_active_learning=False):
    param_path = "params/"
    log.info("Training started")
    device = load_device()
    set_seeds()
    trainset, testset = load_dataset(dataset, model)
    train_idx_df, val_idx_df = load_indices(path=param_path, dataset=trainset)
    net = load_model(model, device)
    optimizer, scheduler = load_modules(net.parameters())
    initial_dict, optim_dict, sched_dict = load_base_dicts(net, optimizer, scheduler, param_path)
    
    df = run_learning(net, device, optimizer, scheduler, 
                      trainset, train_idx_df, val_idx_df, testset, 
                      initial_dict, optim_dict, sched_dict, 
                      epochs=epochs, sub_epochs=sub_epochs, active_learning=is_active_learning)
    
    df.to_csv(f"results/{model}_{dataset}_undersampled.csv")

#TODO: run all experiments

if __name__ == "__main__":
    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    log.basicConfig(filename='./logs/vgg16_cifar_no_schedule.log', filemode='a+', format=FORMAT, level=log.DEBUG)
    models = ["EFFNETV2S","VGG16"]
    datasets = ["CIFAR10", "FashionMNIST", "FastFoodV2"]
    run(model=models[1], dataset=datasets[0], epochs=5, sub_epochs=30, is_active_learning=False) 
    



