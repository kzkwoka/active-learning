import torchvision.transforms as transforms
import torchvision.datasets
import logging as log

def create_transforms(model_name, use_repeat=False):
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

def load_cifar(model_name):
    transform = create_transforms(model_name)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    log.info("Loaded CIFAR10")
    return trainset, testset

def load_fashion_mnist():
    transform = create_transforms(use_repeat=True)
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
    log.info("Loaded FashionMNIST")
    return trainset, testset

def load_fast_food():
    transform = create_transforms()
    
    trainset = torchvision.datasets.ImageFolder(root='data/FastFoodV2/Train', transform=transform)
    # validset =  torchvision.datasets.ImageFolder(root='data/FastFoodV2/Valid', transform=transform)
    testset = torchvision.datasets.ImageFolder(root='data/FastFoodV2/Test', transform=transform)
    log.info("Loaded FastFood V2")
    return trainset, testset