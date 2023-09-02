import torch
import torchvision
import torchvision.transforms as transforms

# transform = transforms.Compose([
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# torch.save(transform, "transform.pth")

# loaded = torch.load("transform.pth")
# print(loaded)

transforms = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT.transforms
print(transforms())
