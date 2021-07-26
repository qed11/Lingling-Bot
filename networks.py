import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d, Conv3d
import torchvision.models
# possible image recognition baseline models to use
alexnet = torchvision.models.alexnet(pretrained=True)
inception = torchvision.models.inception.inception_v3(pretrained=True)
vgg16 = torchvision.models.vgg.vgg16(pretrained=True)
vgg19 = torchvision.models.vgg.vgg19(pretrained=True)
resnet18 = torchvision.models.resnet.resnet18(pretrained=True)
resnet152 = torchvision.models.resnet.resnet152(pretrained=True)

class CNN(nn.Module):
    def __init__(self, name = 'CNN'):
        super(CNN, self).__init__()
        self.name = name
        self.conv1 = nn.Conv2d(1, 5, 3, padding = 1) #128*128*3 - > 128*128*5
        self.pool = nn.MaxPool2d(2, 2) #128*128*10 - 64*64*10 - 32*32*40
        self.conv2 = nn.Conv2d(5, 10, 3, padding = 1) #128*128*5 -> 128*128*10
        self.conv3 = nn.Conv2d(10,20,3, padding = 1) #64*64*10 -> 64*64*20
        self.conv4 = nn.Conv2d(20,40,3, padding = 1) #64*64*20 -> 64*64*40
        self.fc1 = nn.Linear(32*32*40, 300)
        self.fc2 = nn.Linear(300, 64)
        self.fc3 = nn.Linear(64,20)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 32*32*40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, name = 'auto'):
        super(Autoencoder, self).__init__()
        self.name = name
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding = 1), 
            nn.Conv2d(16, 32, 3, padding = 1),
            nn.Conv2d(32, 64, 3, padding = 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, padding = 1, output_padding=1),
            nn.ConvTranspose2d(32, 16, 3, padding = 1, output_padding=1),
            nn.ConvTranspose2d(16, 1, 3, padding = 1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
