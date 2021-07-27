import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from PIL import Image
import librosa as lb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.conv import Conv2d, Conv3d
import torchvision.models
import torch.utils.data as dt
import torchvision.transforms as transforms

os.chdir("/Users/sarinaxi/Desktop/Lingling-Bot")
from networks import CNN

#Set all the seeds to ensure reproducability
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

def load_data(name, bs, hilbert = True):
    '''
    load the data from the files
    '''
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = #'/Users/sarinaxi/Desktop/Lingling-Bot/Data/'+ n +'/features/'
    name = path + name

    # modify according to how data is stored
    if label == True:
        train_name = '_train_labels_duplicates'
        val_name = '_val_labels'
        test_name = '_test_labels'
    else:
        train_name = '_train_duplicates'
        val_name = '_val'
        test_name = '_test'
    trainset = torchvision.datasets.DatasetFolder(name + train_name, loader = torch.load, extensions = ('.tensor'))
    valset = torchvision.datasets.DatasetFolder(name + val_name, loader = torch.load, extensions = ('.tensor'))
    testset = torchvision.datasets.DatasetFolder(name + test_name, loader = torch.load, extensions = ('.tensor'))

    # load data in and return
    trainload = dt.DataLoader(trainset, batch_size = bs, shuffle = True)
    valload = dt.DataLoader(valset, batch_size = bs, shuffle = True)
    testload = dt.DataLoader(testset, batch_size = bs, shuffle = True)

    return trainload, valload, testload

def get_accuracy(model, loader):
    correct = 0
    total = 0
    for feature, label in loader:
        # run on GPU if possible
        if torch.cuda.is_available():
            features = feature[0].squeeze().cuda()
            labels = feature[1].squeeze().cuda()
        else:
            features = feature[0].squeeze()
            labels = feature[1].squeeze()
        outputs = model(features)
        # find the max predictive score
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == labels[i][j]:
                    correct += 1
                total += 1
    return correct / total

def training(model = SimpleCNN(), bs = 27, ne = 1, lr = 0.001, hilbert = True):
    '''
    train the data
    '''
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # load in data and create accuracy arrays
    train_loader, val_loader, test_loader = load_data("model",bs, hilbert)
    train_loss, train_acc, val_acc, iters = [], [], [], []

    # Training
    start_time = time.time()
    i = 0
    print ("Training Started...")
    for epoch in range(ne):
        lo = 0
        j = 0
        for feature, label in iter(train_loader):
            # Run on GPU if possible
            if torch.cuda.is_available():
                features = feature[0].squeeze().cuda()
                labels = feature[1].squeeze().cuda()
            else:
                features = feature[0].squeeze()
                labels = feature[1].squeeze()
            output = model(features)           # forward pass
            loss = criterion(output, labels) # compute loss
            loss.backward()                  # backward pass
            optimizer.step()                 # update parameter
            optimizer.zero_grad()            # clean up
            lo += loss
            j += 1
        iters.append(i)
        i+=1
        train_loss.append(float(lo)/bs/j)           # compute loss
        train_acc.append(get_accuracy(model, train_loader)) # compute train_acc
        val_acc.append(get_accuracy(model, val_loader))   # compute val_acc
        print("Epoch: " + str(epoch) + ', train acc: ' + str(train_acc[-1]) + ', train loss: ' + str(float(train_loss[-1])) + ', valid acc: ' + str(val_acc[-1]))
        if hilbert == True:
            n = "Hilbert"
        else:
            n = "Spectrogram"
        model_path = #"/Users/sarinaxi/Desktop/Lingling-Bot/Data/" + n + "/features/{4}_models/model_customlabel_{0}_bs{1}_lr{2}_epoch{3}".format(model.name, bs, lr, ne, transfer_name)
        torch.save(model.state_dict(), model_path)

    print('Finished Training')
    end_time = time.time()
    elapsed = end_time - start_time
    print("Total time elapsed: " + str(elapsed/60/60) + " hours.")
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    return iters, train_loss, train_acc, val_acc, model.name, bs, lr, ne

def plot_acc_loss(iters, losses, train_acc, val_acc, name, bs, lr, ne, hilbert = True):
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = #"/Users/sarinaxi/Desktop/Lingling-Bot/Data/" + n + "/features/"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.plot(iters, losses, bs, label = 'Train')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(min(losses), max(losses))
    ax1.set_title('Loss Training')

    ax2.plot(iters, train_acc, label="Train")
    ax2.plot(iters, val_acc, label="Validation")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    plt.tight_layout()
    plt.legend()
    plt.savefig("{5}{4}_models/{0}_bs{1}_lr{2}_epoch{3}.png".format(name, bs, lr, ne, transfer_name, path))

use_cuda = True
model = CNN(kernel_size = [3,2], input = 512)
if use_cuda and torch.cuda.is_available():
    model.cuda()
iters, train_loss, train_acc, val_acc, name, bs, lr, ne = training( model, 64, 600, 0.0001, True)
plot_acc_loss(iters, train_loss, train_acc, val_acc, name + "customsoftmargin", bs, lr, ne, True)