from cuda_setup import load_device, set_seeds
from image_datasets import load_cifar
from loading_models import load_modules, load_vgg16
from loading_params import load_base_dicts, load_indices
from training import run_learning

def run():
    param_path = "params/"
    device = load_device()
    set_seeds()
    trainset, testset = load_cifar()
    train_idx_df, val_idx_df = load_indices(path=param_path, dataset=trainset)
    net = load_vgg16(device)
    optimizer, loss_module, scheduler = load_modules(net.parameters())
    initial_dict, optim_dict, sched_dict = load_base_dicts(net, optimizer, scheduler, param_path)
    
    df = run_learning(net, device, optimizer, scheduler, loss_module,
                   trainset, train_idx_df, val_idx_df, testset, 
                   initial_dict, optim_dict, sched_dict, epochs=2, sub_epochs = 3)
    df.to_csv(f"results/vgg16_cifar.csv")
#TODO: run all experiments
#TODO: transform dataset

if __name__ == "__main__":
    run() 
    



