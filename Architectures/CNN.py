import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=(5,5),stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(5,5),stride=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.maxpool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=1)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),stride=(2,2),padding=(1,1))
        self.batch2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(64*2*2, num_classes))

    def forward(self, x):           # [250, 1, 28, 28]
        x = self.conv1(x)           # [250, 32, 24, 24]
        x = self.relu(x)
        x = self.conv2(x)           # [250, 32, 20, 20]
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [250, 32, 10, 10]
        x = self.dropout(x)
        x = self.conv3(x)           # [250, 64, 8, 8]
        x = self.relu(x)
        x = self.conv4(x)              # [250, 64, 4, 4]
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)         # [250, 64, 2, 2]
        x = self.dropout(x)
        out = self.classifier(x)
        return out
