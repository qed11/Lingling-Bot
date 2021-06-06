import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3, padding = 1) #224*224*3 - > 224*224*5
        self.pool = nn.MaxPool2d(2, 2) #112*112*5 - 56*56*10 - 28*28*20
        self.conv2 = nn.Conv2d(5, 10, 3, padding = 1) #112*112*10 -> 112*112*10
        self.conv3 = nn.Conv2d(10,20,3, padding = 1) #56*56*10 -> 56*56*20
        self.conv4 = nn.Conv2d(20,40,3, padding = 1) #28*28*20 -> 28*28*40
        self.fc1 = nn.Linear(14*14*40, 300)
        self.fc2 = nn.Linear(300, 64)
        self.fc3 = nn.Linear(64,9)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 14*14*40)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
