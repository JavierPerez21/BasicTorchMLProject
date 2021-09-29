import torch
import torchvision
import os
from Dataset_generator.tarimagefolder import *



tinyimagenet_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

def get_tinyimagenet_dataset(path):
    if os.path.isdir(path):
        dataset = torchvision.datasets.ImageFolder(path, transform=tinyimagenet_transform)
    else:
        dataset = TarImageFolder(path, transform=tinyimagenet_transform)
    return dataset
