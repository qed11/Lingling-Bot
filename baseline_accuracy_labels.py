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
from preprocessor import hilbert_data, spectrogram_data

# import pretrained models
alexnet = torchvision.models.alexnet(pretrained=True)
vgg16 = torchvision.models.vgg.vgg16(pretrained=True)

#Set all the seeds to ensure reproducability
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

## Functions to save and create labelled dataset
def save_data(win_len = 4096, hilbert = True, save_name = None):
    """
    sort datasets for the labelled data
    win_len is the how many samples are taken from a music clip
    hilbert indiciates whether the data is mapped with hilbert curves or not
    set_percent indicates the % of test/validation
    save_name is the name used to save the datasets

    returns the save_path of the saved data
    """
    dataset = None
    labels = None
    # old_dir is where this file is in: ".../Lingling-Bot/"
    old_dir = os.getcwd()
    labelled_dir = old_dir + "/Downloads/Test/piano/" #Audios/labelled2/"
    save_path = old_dir + "/Data"
    for file in os.listdir(labelled_dir):
        # For each .wav file in the downloaded path
        if file.endswith(".wav"):
            file = labelled_dir + file
            # If hilbert is required, save the .wav files as hilbert curve data
            if hilbert:
                data, label = hilbert_data(file, win_len = win_len)
                # Convert data and labels to tensors
                data, label = torch.Tensor(list(data)), torch.Tensor(list(label))
                if dataset == None:
                    dataset = data
                    labels = label
                else:
                    dataset = torch.cat((dataset, data))
                    labels = torch.cat((labels, label))
            # Else, save the .wav files as spectrogram data
            else:
                data, label = spectrogram_data(file, win_len = win_len)
                # Convert data and labels to tensors
                data, label = torch.Tensor(list(data)), torch.Tensor(list(label))
                if dataset == None:
                    dataset = data
                    labels = label
                else:
                    if data.shape[1] == 128 and data.shape[2] == 128:
                        dataset = torch.cat((dataset, data))
                        labels = torch.cat((labels, label))
                    else:
                        print("skip")
                        print(data.shape)

    if hilbert:
        save_path = save_path + "/Hilbert/labelled/" + save_name + '.pt'
    else:
        save_path = save_path + "/Spectrogram/labelled/" + save_name + '.pt'
    full_set = dt.TensorDataset(dataset, labels)
    torch.save(full_set, save_path)
    return save_path

def get_data(save_path = None, set_percent = 0.1, bs = 1):
    '''
    get the data saves at save_path and return the training, validation, and testing
    '''
    # load data from the path
    load = torch.load(save_path)
    set_size = int(set_percent*len(load))
    training, validation, testing = dt.random_split(load, [len(load) - 2*set_size, set_size, set_size])
    train = dt.DataLoader(training, batch_size = bs, shuffle=True)
    val = dt.DataLoader(validation, batch_size = bs, shuffle=True)
    test = dt.DataLoader(testing, batch_size = bs, shuffle=True)
    return train, val, test

## Store  and loadHilbert/Spectrogram Data
save_path = save_data(win_len = 4096, hilbert = True, save_name = "piano")
##
training_hilb, validation_hilb, testing_hilb = get_data(save_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/labelled/piano.pt")
#training_spec, validation_spec, testing_spec = get_data(save_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Spectrogram/labelled/labelled_spectrogram_new.pt")
## Save Features Function
# save the custom labels
def save_features_labels(loader, size, model, name, hilbert = True):
    # the label of instruments
    label_dic = ['VLN', 'VLA', 'CEL', 'DBS', 'FLT', 'CLT', 'OBO', 'BSN', 'TPT', 'FHN', 'TBN', 'TUB', 'PNO', 'PER']

    # the path of where to store the features
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/"+ n +"/features2/" + name + "/"
    # define transformations
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(size), # make sure the size is right
        transforms.ToTensor(),
    ])
    # create a pre-trained model folder if not exist
    if not os.path.isdir(path):
        os.mkdir(path)

    # store the features
    start = time.time()
    print("Start making features......")
    i = 0
    for images, label in loader:
        # the following are transformations applied to the image to work with alex net
        images = transformations(images)
        images = torch.stack((images, images, images))
        images = np.transpose(images, [1,0,2,3])
        # get the features
        features = model.features(images)
        features = torch.from_numpy(features.detach().numpy())
        # get the folder names for the features of different instruments
        name = ''
        for j in range(len(label.squeeze())):
            if int(label.squeeze()[j].item()) == 1:
                name = str(name) + str(label_dic[j]) + "_"

        # if the label is all 0, then skip over the file
        if name == '':
            continue
        folder = path + name + "/"

        # make the folders and save the features
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save((features,label), folder + str(i) + '.tensor')
        i += 1

    end = time.time()
    diff = end-start
    print("Complete creating features, took " + str(diff/60) + " minutes.")

## Save alexnet features for hilbert mapped
save_features_labels(training_hilb, 224, alexnet, "alexnet_train_labels", hilbert = True)
save_features_labels(validation_hilb, 224, alexnet, "alexnet_val_labels", hilbert = True)
save_features_labels(testing_hilb, 224, alexnet, "alexnet_test_labels", hilbert = True)

## Save vgg16 features for hilbert mapped
save_features_labels(training_hilb, 224, vgg16, "vgg16_train_labels", hilbert = True)
save_features_labels(validation_hilb, 224, vgg16, "vgg16_val_labels", hilbert = True)
save_features_labels(testing_hilb, 224, vgg16, "vgg16_test_labels", hilbert = True)

## get number of features
def nums(name):
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/autoCNN2/" + name + "/"
    folders = []
    num_items = []
    items = 0
    # iterate through all the folders within the training dataset folder
    for folder in os.listdir(path):
        if os.path.isdir(path + folder + "/"):
            # get all the folders
            folders.append(folder)
            i = 0
            # count the number of items in the folder
            for item in os.listdir(path + folder + "/"):
                i += 1
            # store the number of items in the folder
            num_items.append(i)
            items += num_items[-1]
    return items

print(nums("train_labels"))
print(nums("train_labels_duplicates"))
print(nums("val_labels"))
print(nums("test_labels"))

## Balacing the training dataset
import shutil

def balance_training_set(name, hilbert = True):
    folders = []
    num_items = []
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/"+ n +"/features2/" + name + "/"

    # iterate through all the folders within the training dataset folder
    for folder in os.listdir(path):
        if os.path.isdir(path + folder + "/"):
            # get all the folders
            folders.append(folder)
            i = 0
            # count the number of items in the folder
            for item in os.listdir(path + folder + "/"):
                i += 1
            # store the number of items in the folder
            num_items.append(i)
    # find the max value of items in a folder
    max_nums = max(num_items)
    # find out how much you need to increase the amount of items in the other folders
    ratios = []
    for i in num_items:
        ratios.append(round(max_nums/i))

    # make folder for duplicates
    new_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/"+ n +"/features2/" + name + "_duplicates/"
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    # make duplicates
    k = 0
    num_items_new = []
    for folder in os.listdir(path):
        if os.path.isdir(path + folder + "/"):
            # iterate through items in the folder
            for item in os.listdir(path + folder + "/"):
                name = path + folder + '/' + item
                # new folder for fuplicates
                new_name = new_path + folder + "/"
                if not os.path.isdir(new_name):
                    os.mkdir(new_name)
                for j in range(ratios[k]):
                    shutil.copy(name, new_name + item[:-7]+ "_" + str(j) + ".tensor")
        else:
            k -= 1
        k += 1

    # get the number of items in the duplicate folders to check
    num_items_new = []
    for folder in os.listdir(new_path):
        if os.path.isdir(new_path + folder + '/'):
            i = 0
            # count the number of items in the folder
            for item in os.listdir(new_path + folder + "/"):
                i += 1
            # store the number of items in the folder
            num_items_new.append(i)
    return folders, num_items, ratios, num_items_new


## create new data folder for more balanced dataset
folders, num_items, ratios, new = balance_training_set("alexnet_train_labels", hilbert = True )

##
folders, num_items, ratios, new = balance_training_set("vgg16_train_labels", hilbert = True)

## Load the data from the balanced datasets
def load_data(name, bs, label = False, hilbert = False):
    '''
    load the data from the files
    '''
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = '/Users/sarinaxi/Desktop/Lingling-Bot/Data/'+ n +'/features/'
    name = path + name

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


# train_alex_label, val_alex_label, test_alex_label = load_data("alexnet", 1, True)
# for i, j in train_alex_label:
#     print(len(i[1][0][0]))
#     break


## Create Simple Model CNN for Alexnet and vgg16
class SimpleCNN(nn.Module):
    def __init__(self, kernel_size = [2,2], input = 256):
        super(SimpleCNN, self).__init__()
        self.name = "SimpleCNN"

        # 2 convolution layers
        self.conv1 = nn.Conv2d(input, 49, kernel_size[0])
        self.conv2 = nn.Conv2d(49, 10, kernel_size[1])

        # Fully connected layers, hidden unit of 32
        self.fc1 = nn.Linear(10*4*4, 32)
        self.fc2 = nn.Linear(32, 20) # 14 classifications

    def forward(self, img):
        #print(img.shape)
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = x.view(-1, 10*4*4)

        # use relu as activation function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x[0])
        m = nn.Sigmoid()
        x = m(x)
        #print(x)
        return x

## Training Code (not working yet)
# get the accuracy of the model
def get_accuracy(model, loader):
    correct = 0
    total = 0
    conf_matrix = np.zeros([20, 2, 2])
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
                if outputs[i][j] > 0.5:
                    outputs[i][j] = 1
                else:
                    outputs[i][j] = 0
                if (outputs[i][j] == 1) and (labels[i][j] == 1): #True Positive
                    correct += 1
                    conf_matrix[j, 0, 0] += 1
                elif (outputs[i][j] == 1) and (labels[i][j] == 0): #False Positive
                    conf_matrix[j, 0, 1] += 1
                elif (outputs[i][j] == 0) and (labels[i][j] == 1): #False Negative
                    conf_matrix[j, 1, 0] += 1
                elif (outputs[i][j] == 0) and (labels[i][j] == 0): #True Negative
                    correct += 1
                    conf_matrix[j, 1, 1] += 1
                #else:
                #    #print("uh oh")
                total += 1
    return correct / total, conf_matrix/total * 20
def get_loss(model, loader, criterion, bs):
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    i = 0
    for feature, label in loader:
        # run on GPU if possible
        if torch.cuda.is_available():
            features = feature[0].squeeze().cuda()
            labels = feature[1].squeeze().cuda()
        else:
            features = feature[0].squeeze()
            labels = feature[1].squeeze()
        outputs = model(features)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        i += 1
    loss = float(total_loss) / (i + 1) /bs

    return loss

def training(transfer_name = "alexnet", model = SimpleCNN(), bs = 27, ne = 1, lr = 0.001, hilbert = True):
    '''
    train the data
    transfer_name is "alexnet" or "vgg16"
    '''
    # use cross entropy loss for multi classification and adam optimizer

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # load in data and create accuracy arrays
    train_loader, val_loader, test_loader = load_data(transfer_name, bs, True, hilbert)
    train_loss, train_acc, val_acc, val_loss, iters = [], [], [], [], []

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
        train_loss.append(float(lo)/j/bs)           # compute loss
        train_acc.append(get_accuracy(model, train_loader)[0]) # compute train_acc
        val_acc.append(get_accuracy(model, val_loader)[0])   # compute val_acc
        val_loss.append(get_loss(model, val_loader, criterion, bs))
        print("Epoch: " + str(epoch) + ', train acc: ' + str(train_acc[-1]) + ', train loss: ' + str(float(train_loss[-1])) + ', valid acc: ' + str(val_acc[-1]) + ", val loss: " + str(float(val_loss[-1])))
        if hilbert == True:
            n = "Hilbert"
        else:
            n = "Spectrogram"
        model_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/" + n + "/features/{4}_models/model_{0}_bs{1}_lr{2}_epoch{3}".format(model.name, bs, lr, ne, transfer_name)
        torch.save(model.state_dict(), model_path)

    print('Finished Training')
    end_time = time.time()
    elapsed = end_time - start_time
    print("Total time elapsed: " + str(elapsed/60/60) + " hours.")
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    return iters, train_loss, train_acc, val_acc, val_loss, model.name, bs, lr, ne, transfer_name

def plot_acc_loss(iters, losses, train_acc, val_acc, val_loss, name, bs, lr, ne, transfer_name, hilbert = True):
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/" + n + "/features/"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.plot(iters, losses, bs, label = 'Train')
    #ax1.plot(iters, val_loss, bs, label = 'Validation')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(min(losses + val_loss), max(losses + val_loss))
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
model = SimpleCNN(kernel_size = [3,2], input = 512)
if use_cuda and torch.cuda.is_available():
    model.cuda()

## training alexnet
iters, train_loss, train_acc, val_acc, val_loss, name, bs, lr, ne, transfer_name = training('alexnet', model, 64, 10, 0.0001, False)
plot_acc_loss(iters, train_loss, train_acc, val_acc, val_loss, name, bs, lr, ne, transfer_name, False)

## training VGG16
vgg_iters, vgg_train_loss, vgg_train_acc, vgg_val_acc, vgg_val_loss, vgg_name, vgg_bs, vgg_lr, vgg_ne, vgg_transfer_name = training('vgg16', model, 64, 10, 0.0001, False)
plot_acc_loss(vgg_iters, vgg_train_loss,vgg_train_acc, vgg_val_acc, vgg_val_loss, vgg_name + "new", vgg_bs, vgg_lr, vgg_ne, vgg_transfer_name,False)

## get test accuracy
# variables
bs = 64
ne = 10
lr = 0.0001
transfer_name = "vgg16"

#model_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features2/{4}_models/model_customlabel_{0}_bs{1}_lr{2}_epoch{3}".format("SimpleCNN", bs, lr, ne, transfer_name)
model_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Spectrogram/features/vgg16_models/model_SimpleCNN_bs64_lr0.0001_epoch10"
state = torch.load(model_path)
use_cuda = True
model = SimpleCNN(kernel_size = [3,2], input = 512) # [3, 2], input = 512
if use_cuda and torch.cuda.is_available():
    model.cuda()
model.load_state_dict(state)

train_loader, val_loader, test_loader = load_data(transfer_name, bs, True)

# 0.7338408949658173 for softmargin1 (64, 100, 0.001)
# 0.808701565568676 (64,150,0.001)
# 0.7158172778123058 for (64, 10, 0.001)

test_acc = get_accuracy(model, test_loader)
print("test accuracy:", test_acc[0])
print("Confusion Matricies:")
label_dic = ['VLN', 'VLA', 'CEL', 'DBS', 'FLT', 'CLT', 'OBO', 'BSN',  'TPT', 'FHN', 'TBN', 'TUB', 'PNO', 'PER']
for i in range(len(test_acc[1])):
    print(label_dic[i])
    print(test_acc[1][i])
