from cuda_setup import load_device, set_seeds
from image_datasets import load_cifar
from loading_models import load_modules, load_vgg16
from loading_params import load_indices

def run():
    device = load_device()
    set_seeds()
    trainset, testset = load_cifar()
    train_idx_df, val_idx_df = load_indices(path="params/", dataset=trainset)
    net = load_vgg16(device)
    optimizer, loss_module, scheduler = load_modules(net.parameters())

if __name__ == "__main__":
    run()