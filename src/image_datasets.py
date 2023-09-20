import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets
import logging as log


def load_dataset(dataset, model):
    ["CIFAR10", "FashionMNIST", "FastFoodV2", "PlantVillage"]
    if dataset == "CIFAR10":
        return load_cifar(model)
    elif dataset == "FashionMNIST":
        return load_fashion_mnist(model)
    elif dataset == "FastFoodV2":
        return load_fast_food(model)
    elif dataset == "PlantVillage":
        return load_plant_village(model)
    else:
        raise Exception("Invalid dataset")


def load_cifar(model_name):
    transform = create_transforms(model_name)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    
    id_path = "params/cifar"
    if not os.path.exists(id_path):
        os.makedirs(id_path)
    id_path += "/undersampled_indices.pt"
    trainset = undersample_dataset(trainset, id_path)
    log.info("Loaded CIFAR10")
    return trainset, testset

def load_fashion_mnist(model_name):
    transform = create_transforms(model_name, use_repeat=True)
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
    id_path = "params/fashion"
    if not os.path.exists(id_path):
        os.makedirs(id_path)
    id_path += "/undersampled_indices.pt"
    trainset = undersample_dataset(trainset, id_path)
    log.info("Loaded FashionMNIST")
    return trainset, testset

def load_fast_food(model_name):
    transform = create_transforms(model_name)
    
    trainset = torchvision.datasets.ImageFolder(root='data/Fast Food Classification V2/Train', transform=transform)
    # validset =  torchvision.datasets.ImageFolder(root='data/FastFoodV2/Valid', transform=transform)
    testset = torchvision.datasets.ImageFolder(root='data/Fast Food Classification V2/Test', transform=transform)
    id_path = "params/food"
    if not os.path.exists(id_path):
        os.makedirs(id_path)
    id_path += "/undersampled_indices.pt"
    trainset = undersample_dataset(trainset, id_path)
    log.info("Loaded FastFood V2")
    return trainset, testset

def load_plant_village(model_name):
    transform = create_transforms(model_name)
    
    trainset = torchvision.datasets.ImageFolder(root='data/PlantVillage/dataset/train', transform=transform)
    # validset =  torchvision.datasets.ImageFolder(root='data/FastFoodV2/Valid', transform=transform)
    testset = torchvision.datasets.ImageFolder(root='data/PlantVillage/dataset/test', transform=transform)
    # id_path = "params/food"
    # if not os.path.exists(id_path):
    #     os.makedirs(id_path)
    log.info("Dataset already imbalanced")
    # id_path += "/undersampled_indices.pt"
    # trainset = undersample_dataset(trainset, id_path)
    log.info("Loaded PlantVillage")
    return trainset, testset

def undersample_dataset(dataset, id_path):
    targets = np.array(dataset.targets)
    if os.path.exists(id_path):
        imbal_class_indices = torch.load(id_path)
        log.info(f"Loaded imbalanced indices from {id_path}")
    else:
        classes, class_counts = np.unique(targets, return_counts=True)
        n = len(classes)
        weights = load_weights_for_undersampling(n)
        # imbal_class_counts = np.multiply(weights, class_counts).astype(int)
        imbal_class_counts = np.multiply(weights, sum(class_counts)).astype(int)
        log.info(f"Using {imbal_class_counts} of consecutive class samples")
        class_indices = [np.where(targets == i)[0] for i in range(n)]
        imbal_class_indices = [class_idx[:class_count] for class_idx, class_count in zip(class_indices, imbal_class_counts)]
        imbal_class_indices = np.hstack(imbal_class_indices)
        torch.save(imbal_class_indices, id_path)
        log.info(f"Saved imbalanced indices to {id_path}")

    # Set target and data to dataset
    dataset.targets = targets[imbal_class_indices]
    try:
        dataset.data = dataset.data[imbal_class_indices]
    except AttributeError:
        dataset.samples = [dataset.samples[i] for i in imbal_class_indices]
    return dataset

def load_weights_for_undersampling(n):
    r = np.random.randint(1,n*10,n)
    return [ i/sum(r) for i in r ]

def create_transforms(model_name, use_repeat=False):
    #TODO: fill use_repeat for fashion mnist
    if model_name == "VGG16":
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        #torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms
    elif model_name == "EFFNETV2S":
        transform = transforms.Compose([
                    transforms.Resize(384),
                    transforms.CenterCrop(384),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    elif model_name == "OWN":
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(256),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])
    else:
        if use_repeat:
            transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        else:
            transform = transforms.Compose([   
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform




    