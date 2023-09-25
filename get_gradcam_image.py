import cv2
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageNet
from utils import get_augmentation

from models.gradcam import GradCam


# Here you need to write a path to model
path_to_model = "/content/CIFAR_model_CIFAR10.pt"
model = torch.load(path_to_model)

# Here choose dataset

dsname = "CIFAR10"
train = False

transformer_data = get_augmentation(train=False, do_hflip=False, do_vflip=False)
if dsname == "MNIST":
    dataset = MNIST(dsname, train=train, download=True, transform=transformer_data)
elif dsname == "CIFAR10":
    dataset = CIFAR10(dsname, train=train, download=True, transform=transformer_data)
elif dsname == "CIFAR100":
    dataset = CIFAR100(dsname, train=train, download=True, transform=transformer_data)
elif dsname == "ImageNet":
    dataset = ImageNet(dsname, train=train, download=True, transform=transformer_data)


img, label = dataset[0]

res = model.compute_gradcam(img.unsqueeze(0), ["backbone", "blocks", "1"])
res = res.squeeze().detach().numpy()
print(res)