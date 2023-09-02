
import torch
import torchvision
import torch.nn as nn
import torchvision.models.vgg
import logging as log

def load_model(model_name, device, n):
    if model_name == "VGG16":
        return load_vgg16(device, n)
    elif model_name == "EFFNETV2S":
        return load_effnet_v2s(device, n)
    elif model_name == "OWN":
        return load_own_model(device, n)
    else:
        log.error(f"Invalid model name passed {model_name}")
        return
    
class OWNModel(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc5 = nn.Linear(524288, 32768)
        self.act5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.5)
        
        self.fc6 = nn.Linear(32768, 512)
        self.act6 = nn.ReLU()
        self.drop6 = nn.Dropout(0.5)

        self.fc7 = nn.Linear(512, n_classes)

    def forward(self, x):
        # input 3x256x256, output 64x256x256
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        # input 64x256x256, output 64x256x256
        x = self.act2(self.conv2(x))
        # input 64x256x256, output 64x128x128
        x = self.pool2(x)
        
        # input 64x128x128, output 128x128x128
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        # input 128x128x128, output 128x128x128
        x = self.act4(self.conv4(x))
        # input 128x128x128, output 128x64x64
        x = self.pool4(x)
        
        # input 128x64x64, output 524288
        x = self.flat(x)
        # input 524288, output 32768
        x = self.act5(self.fc5(x))
        x = self.drop5(x)
        # input 32768, output 512
        x = self.act6(self.fc6(x))
        x = self.drop6(x)
        # input 512, output 10
        x = self.fc7(x)
        return x
        

def load_own_model(device: torch.device, n: int=10) -> OWNModel:
    """load created net model for n classes

    Args:
        device (torch.device): device to store the model
        n (int, optional): number of classes for the model to predict. Defaults to 10.

    Returns:
        _type_: loaded model
    """ 
    net = OWNModel(n)
    log.info(f"Loaded OWN model with {n} classes")
    net.to(device)
    return net

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
    return optimizer, scheduler