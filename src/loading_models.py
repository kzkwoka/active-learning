
import torch
import torchvision
import torch.nn as nn
import torchvision.models.vgg

def load_vgg16(device: torch.device, n: int=10) -> torchvision.models.vgg.VGG:
    """load pretrained VGG16 net for n classes

    Args:
        device (torch.device): device to store the model
        n (int, optional): number of classes for the model to predict. Defaults to 10.

    Returns:
        _type_: loaded model
    """    
    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT) # get vgg16 model with pretrained weights

    # vgg16.classifier[6].out_features = n
    
    # freeze convolution weights
    for param in vgg16.features.parameters():
        param.requires_grad = False
        
    # change the number of classes in the last layer
    input_lastLayer = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(input_lastLayer,n)
    vgg16.to(device)
    return vgg16

#TODO: add more models

def load_modules(params):
    optimizer = torch.optim.Adam(params)
    loss_module = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    return optimizer, loss_module, scheduler