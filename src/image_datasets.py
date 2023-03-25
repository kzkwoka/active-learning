import torchvision.transforms as transforms
import torchvision.datasets

def create_transforms(use_repeat=False):
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

def load_cifar():
    transform = create_transforms()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    return trainset, testset

def load_fashion_mnist():
    transform = create_transforms(use_repeat=True)
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
    return trainset, testset

def load_fast_food():
    transform = create_transforms()
    
    trainset = torchvision.datasets.ImageFolder(root='data/FastFoodV2/Train', transform=transform)
    # validset =  torchvision.datasets.ImageFolder(root='data/FastFoodV2/Valid', transform=transform)
    testset = torchvision.datasets.ImageFolder(root='data/FastFoodV2/Test', transform=transform)
    return trainset, testset