
import torch
import torchvision
import torch.nn as nn
import torchvision.models.vgg
import logging as log

def load_model(model_name, device):
    if model_name == "VGG16":
        return load_vgg16(device)
    elif model_name == "EFFNETV2S":
        return load_effnet_v2s(device)
    else:
        log.error(f"Invalid model name passed {model_name}")
        return
    

def load_vgg16(device: torch.device, n: int=10) -> torchvision.models.vgg.VGG:
    """load pretrained VGG16 net for n classes

    Args:
        device (torch.device): device to store the model
        n (int, optional): number of classes for the model to predict. Defaults to 10.

    Returns:
        _type_: loaded model
    """    
    vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT) # get vgg16 model with pretrained weights
    log.info("Loaded VGG16 with default pretrained weights")
    # vgg16.classifier[6].out_features = n
    
    # freeze convolution weights
    for param in vgg16.features.parameters():
        param.requires_grad = False
        
    # change the number of classes in the last layer
    input_lastLayer = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(input_lastLayer,n)
    log.info(f"Classifier modified to {n} classes")
    vgg16.to(device)
    return vgg16


def load_effnet_v2s(device: torch.device, n: int=10) -> torchvision.models.efficientnet.EfficientNet:
    """load pretrained EfficientNet V2 S for n classes

    Args:
        device (torch.device): device to store the model
        n (int, optional): number of classes for the model to predict. Defaults to 10.

    Returns:
        _type_: loaded model
    """    
    effnet = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT) # get vgg16 model with pretrained weights
    log.info("Loaded EfficientNet V2 S with default pretrained weights")
    
    # freeze convolution weights
    for param in effnet.features.parameters():
        param.requires_grad = False
        
    # change the number of classes in the last layer
    input_lastLayer = effnet.classifier[1].in_features
    effnet.classifier[1] = nn.Linear(input_lastLayer,n)
    log.info(f"Classifier modified to {n} classes")
    effnet.to(device)
    return effnet


def load_modules(params):
    optimizer = torch.optim.Adam(params)
    log.info(f"Loaded optimizer ADAM with default parameters")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    log.info(f"Loaded LR scheduler")
    # log.info(f"Not using a LR scheduler")
    return optimizer, None