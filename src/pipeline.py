import logging as log
from cuda_setup import load_device, set_seeds
from image_datasets import load_cifar
from loading_models import load_effnet_v2s, load_model, load_modules, load_vgg16
from loading_params import load_base_dicts, load_indices
from training import run_learning

def run():
    param_path = "params/"
    log.info("Training started")
    device = load_device()
    set_seeds()
    model = "EFFNETV2S"
    trainset, testset = load_cifar(model)
    train_idx_df, val_idx_df = load_indices(path=param_path, dataset=trainset)
    net = load_model(model, device)
    optimizer, loss_module, scheduler = load_modules(net.parameters())
    initial_dict, optim_dict, sched_dict = load_base_dicts(net, optimizer, scheduler, param_path)
    
    df = run_learning(net, device, optimizer, scheduler, loss_module,
                   trainset, train_idx_df, val_idx_df, testset, 
                   initial_dict, optim_dict, sched_dict, epochs=1, sub_epochs=41)
    df.to_csv(f"results/effnet_cifar.csv")

#TODO: run all experiments
#TODO: transform dataset

if __name__ == "__main__":
    FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
    log.basicConfig(filename='./logs/app.log', filemode='a+', format=FORMAT, level=log.DEBUG)
    run() 
    



