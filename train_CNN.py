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

#Set all the seeds to ensure reproducability
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

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
    labelled_dir = old_dir + "/Downloads/Test/horn/" #Audios/labelled2/"
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

## get the data
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

save_path = save_data(win_len = 4096, hilbert = True, save_name = "horn")
training_hilb, validation_hilb, testing_hilb = get_data(save_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/labelled/horn.pt")

## properly save labelled data
def save_with_labels(loader, size, name, hilbert = True):
    # the label of instruments
    label_dic = ['VLN', 'VLA', 'CEL', 'DBS', 'FLT', 'CLT', 'OBO', 'BSN', 'TPT', 'FHN', 'TBN', 'TUB', 'PNO', 'PER']
    # the path of where to store the features
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/"+ n +"/horn/" + name + "/"
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
        images = transformations(images)
        images = images.unsqueeze(0)

        # get folder name
        name = ''
        for j in range(len(label.squeeze())):
            if int(label.squeeze()[j].item()) == 1:
                name = str(name) + str(label_dic[j]) + "_"
        # if the label is all 0, then skip over the file
        if name == '':
            continue
        folder = path + name + "/"

        # make the folders and labelled data
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save((images,label), folder + str(i) + '.tensor')
        i += 1

    end = time.time()
    diff = end-start
    print("Complete creating folders, took " + str(diff/60) + " minutes.")

save_with_labels(training_hilb, 128,  "train_labels")
save_with_labels(validation_hilb, 128, "val_labels")
save_with_labels(testing_hilb, 128,  "test_labels")

## Balacing the training dataset
import shutil

def balance_training_set(name, hilbert = True):
    folders = []
    num_items = []
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/"+ n +"/autoCNN2/" + name + "/"

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
    new_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/"+ n +"/autoCNN2/" + name + "_duplicates/"
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

folders, num_items, ratios, new = balance_training_set("test_labels")
folders, num_items, ratios, new = balance_training_set("val_labels")
folders, num_items, ratios, new = balance_training_set("train_labels")

##
import shutil

def balance(name, hilbert = True):
    folders = []
    num_items = []
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/"+ n +"/autoCNN2/" + name + "/"
    label_dic = ['VLN', 'VLA', 'CEL', 'DBS', 'FLT', 'CLT', 'OBO', 'BSN', 'TPT', 'FHN', 'TBN', 'TUB', 'PNO', 'PER']

    # iterate through all the folders within the training dataset folder
    for folder in os.listdir(path):
        if os.path.isdir(path + folder + "/"):
            for i in label_dic:
                if i in folder:
                    if  not os.path.isdir("/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/CNN/"+ name + "/"+i + "/"):
                        os.mkdir("/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/CNN/"+ name + "/"+i + "/")
                    for j in os.listdir(path + folder + "/"):
                        new_name = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/CNN/"+ name + "/"+i + "/" + j
                        shutil.copy(path + folder + "/" + j, new_name)

def balance2(name, hilbert=True):
    ratio = []
    fold = []
    items = []
    path2 = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/CNN/" + name + "/"
    # iterate through all the folders within the training dataset folder
    for folder in os.listdir(path2):
        if os.path.isdir(path2 + folder + "/"):
            # get all the folders
            fold.append(folder)
            i = 0
            # count the number of items in the folder
            for item in os.listdir(path2 + folder + "/"):
                i += 1
            # store the number of items in the folder

            items.append(i)
    # find the max value of items in a folder
    max_nums = max(items)
    # find out how much you need to increase the amount of items in the other folders
    for i in items:
        ratio.append(round(max_nums/i))

    # make folder for duplicates
    new_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/CNN/" + name + "_duplicates/"
    if not os.path.isdir(new_path):
        os.mkdir(new_path)
    # make duplicates
    k = 0
    for folder in os.listdir(path2):
        if os.path.isdir(path2 + folder + "/"):
            # iterate through items in the folder
            for item in os.listdir(path2 + folder + "/"):
                name = path2 + folder + '/' + item
                # new folder for fuplicates
                new_name = new_path + folder + "/"
                if not os.path.isdir(new_name):
                    os.mkdir(new_name)
                #print(name)
                #print(new_name)
                for j in range(ratio[k]):
                    shutil.copy(name, new_name + item[:-7]+ "_" + str(j) + ".tensor")
        else:
            k -= 1
        k += 1

    return fold, items, ratio

#fold, num_items, ratio, new = balance2("test_labels")
balance("train_labels")
fold, num_items, ratio= balance2("train_labels")
balance("val_labels")
fold, num_items, ratio= balance2("val_labels")

## load in data
def load_data(bs, hilbert = True):
    '''
    load the data from the files
    '''
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = '/Users/sarinaxi/Desktop/Lingling-Bot/Data/'+ n +'/autoCNN2/'
    name = path

    # modify according to how data is stored

    train_name = 'train_labels_duplicates/'
    val_name = 'val_labels/'
    test_name = 'test_labels/'

    trainset = torchvision.datasets.DatasetFolder(name + train_name, loader = torch.load, extensions = ('.tensor'))
    valset = torchvision.datasets.DatasetFolder(name + val_name, loader = torch.load, extensions = ('.tensor'))
    testset = torchvision.datasets.DatasetFolder(name + test_name, loader = torch.load, extensions = ('.tensor'))

    # load data in and return
    trainload = dt.DataLoader(trainset, batch_size = bs, shuffle = True)
    valload = dt.DataLoader(valset, batch_size = bs, shuffle = True)
    testload = dt.DataLoader(testset, batch_size = bs, shuffle = True)

    return trainload, valload, testload

## The model
class CNN2(nn.Module):
    def __init__(self, name = "CNN2"):
        super(CNN2, self).__init__()
        self.name = name
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, stride = 2),
            nn.ReLU()
        )

        # Fully connected layers, hidden unit of 32
        self.fc1 = nn.Linear(128*2*2,32)
        #self.fc2 = nn.Linear(50, 32)
        self.fc2 = nn.Linear(32, 14)
        #self.fc3 = nn.Linear(50, 20) # 20 classifications

    def forward(self, img):
        imgs = torch.reshape(img, [len(img), 1, 128, 128])
        x = self.encoder(imgs)
        x = x.view(-1, 128*2*2)

        # use relu as activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc3(x))
        #print(x[0])
        #m = nn.Sigmoid()
        #x = m(x)
        #print(x[0])
        return x

## training
def get_accuracy(model, loader):
    correct = 0
    total = 0
    conf_matrix = np.zeros([14, 2, 2])
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
                else:
                    print("uh oh")
                total += 1
    return correct / total, conf_matrix/total * 14

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

def training(model = CNN2(), bs = 27, ne = 1, lr = 0.001, hilbert = True):
    '''
    train the data
    '''
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # load in data and create accuracy arrays
    train_loader, val_loader, test_loader = load_data(bs, hilbert)
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
                features = feature[0].squeeze(1).cuda()
                labels = feature[1].squeeze(1).cuda()
            else:
                features = feature[0].squeeze(1)
                labels = feature[1].squeeze(1)
            if len(features) == bs:
                output = model(features)           # forward pass

                loss = criterion(output, labels) # compute loss
                loss.backward()                  # backward pass
                optimizer.step()                 # update parameter
                optimizer.zero_grad()            # clean up
                lo += loss
                j += 1
                #print(output)
                #print(labels)
        iters.append(i)
        i+=1
        train_loss.append(float(lo)/bs/j)           # compute loss
        train_acc.append(get_accuracy(model, train_loader)[0]) # compute train_acc
        val_acc.append(get_accuracy(model, val_loader)[0])   # compute val_acc
        val_loss.append(get_loss(model, val_loader, criterion, bs))
        print("Epoch: " + str(epoch) + ', train acc: ' + str(train_acc[-1]) + ', train loss: ' + str(float(train_loss[-1])) + ', valid acc: ' + str(val_acc[-1]) + ", val_loss: " + str(float(val_loss[-1])))
        if hilbert == True:
            n = "Hilbert"
        else:
            n = "Spectrogram"
        model_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/" + n + "/piano/models/model_3layer50_{0}_bs{1}_lr{2}_epoch{3}".format(model.name, bs, lr, ne)
        torch.save(model.state_dict(), model_path)

    print('Finished Training')
    end_time = time.time()
    elapsed = end_time - start_time
    print("Total time elapsed: " + str(elapsed/60/60) + " hours.")
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    return iters, train_loss, train_acc, val_acc, val_loss, model.name, bs, lr, ne

def plot_acc_loss(iters, losses, train_acc, val_acc, val_loss, name, bs, lr, ne, hilbert = True):
    if hilbert == True:
        n = "Hilbert"
    else:
        n = "Spectrogram"
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/" + n + "/piano/"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.plot(iters, losses, bs, label = 'Train')
    ax1.plot(iters, val_loss, bs, label = 'Validation')
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
    plt.savefig("{4}models/{0}_bs{1}_lr{2}_epoch{3}.png".format(name, bs, lr, ne, path))

use_cuda = True
model = CNN2()
if use_cuda and torch.cuda.is_available():
    model.cuda()
##
iters, train_loss, train_acc, val_acc, val_loss, name, bs, lr, ne = training(model, 64, 10, 0.001, False)
##
plot_acc_loss(iters, train_loss, train_acc, val_acc, val_loss, name + "3layer502", bs, lr, ne, False)

##
bs = 64
ne = 10
lr = 0.0001
#model_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/autoCNN2/models/model_customlabel_{0}_bs{1}_lr{2}_epoch{3}".format("CNN2", bs, lr, ne)
model_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/autoCNN2/models/model_customlabel_CNN2_bs64_lr0.001_epoch10"
state = torch.load(model_path)
use_cuda = True
model = CNN2()
if use_cuda and torch.cuda.is_available():
    model.cuda()
model.load_state_dict(state)

train_loader, val_loader, test_loader = load_data(bs, True)
test_acc = get_accuracy(model, train_loader)

for i in test_loader:
    print(i[0][-1][-1][0].tolist())
    break
print("test accuracy:", test_acc[0]) # 0.7158172778123058 for (64, 10, 0.001)
#print("Confusion Matricies:")
label_dic = ['VLN', 'VLA', 'CEL', 'DBS', 'FLT', 'CLT', 'OBO', 'BSN', 'TPT', 'FHN', 'TBN', 'TUB', 'PNO', 'PER']
result = []

for i in range(len(test_acc[1])):
    print(label_dic[i])
    print(test_acc[1][i])

    # if test_acc[1][i][0][0] > 0.5:
    #     result.append(float(1))
    #
    # elif test_acc[1][i][1][1] > 0.5:
    #     result.append(float(0))
    # else:
    #     result.append(0.5)
#print(result)
