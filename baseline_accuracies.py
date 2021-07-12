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

# Set all the seeds to ensure reproducability
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
    labelled_dir = old_dir + "/Downloads/Audios/try/"
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
                    dataset = torch.cat((dataset, data))
                    labels = torch.cat((labels, label))

    if hilbert:
        save_path = save_path + "/Hilbert/labelled/" + save_name + '.pt'
    else:
        save_path = save_path + "/Spectrogram/labelled/" + save_name + '.pt'
    full_set = dt.TensorDataset(dataset, labels)
    # print(dataset.shape)
    # print(labels.shape)
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

## Store Hilbert Data
# get train, valid, test from labelled data
save_path = save_data(win_len = 4096, hilbert = True, save_name = "labelled_hilbert_4096")

## Store Spectrogram Data
# get train, valid, test from labelled data
save_path_spec = save_data(win_len = 4096, hilbert = False, save_name = "labelled_spectrogram_4096")

## Load data for hilbert and spectrogram
training_hilb, validation_hilb, testing_hilb = get_data(save_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/labelled/labelled_hilbert_4096.pt")
# training_spec, validation_spec, testing_spec = get_data(save_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/labelled/labelled_spectrogram.pt")

## Save Features Function
# this function save the features without the labels
def save_features(loader, size, model, name):
    # the label of instruments
    label_dic = ['VLN', 'VLA', 'CEL', 'DBS', 'HRP', 'PCO', 'FLT', 'CLT', 'OBO', 'EHN', 'BSN', 'BCL', 'CTB', 'TPT', 'FHN', 'TBN', 'TUB', 'PNO', 'HSD', 'PER']
    # the path of where to store the features
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/" + name + "/"
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
        #print(folder)
        # make the folders and save the features
        if not os.path.isdir(folder):
            os.mkdir(folder)
        torch.save(features.squeeze(0), folder + str(i) + '.tensor')
        i += 1
    end = time.time()
    diff = end-start
    print("Complete creating features, took " + str(diff/60) + " minutes.")

## Save alexnet features
save_features(training_hilb, 224, alexnet, "alexnet_train")
save_features(validation_hilb, 224, alexnet, "alexnet_val")
save_features(testing_hilb, 224, alexnet, "alexnet_test")

## Save vgg16 features
save_features(training_hilb, 224, vgg16, "vgg16_train")
save_features(validation_hilb, 224, vgg16, "vgg16_val")
save_features(testing_hilb, 224, vgg16, "vgg16_test")

## get number of features
folders = []
num_items = []
items = 0
name = 'vgg16_train_labels'
path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/" + name + "/"

# iterate through all the folders within the training dataset folder
for folder in os.listdir(path):
    if os.path.isdir(path + folder + "/"):
        # get all the folders
        folders.append(folder)
        i = 0
        # count the number of items in the folder
        for item in os.listdir(path + folder + "/"):
            i += 1
        #k = torch.load(path + folder + '/' + item)
        # store the number of items in the folder
        num_items.append(i)
        #print(k.shape)
        items += num_items[-1]

## Balacing the training dataset
import shutil

def balance_training_set(name):
    folders = []
    num_items = []
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/" + name + "/"

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
    new_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/" + name + "_duplicates/"
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
        #print(k)

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
folders, num_items, ratios, new = balance_training_set("alexnet_train")
folders, num_items, ratios, new = balance_training_set("vgg16_train")

## Load the data from the balanced datasets
def load_data(name, bs, label = False):
    '''
    load the data from the files
    '''
    path = '/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/'
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

## load the loaders for alexnet and vgg16
train_alex, val_alex, test_alex = load_data("alexnet", 1, False)
train_vgg, val_vgg, test_vgg = load_data('vgg16', 1, False)

## Create Simple Model CNN for Alexnet and vgg16
class SimpleCNN(nn.Module):
    def __init__(self, kernel_size = [3,3], input = 256):
        super(SimpleCNN, self).__init__()
        self.name = "SimpleCNN"

        # 2 convolution layers
        self.conv1 = nn.Conv2d(input, 49, kernel_size[0])
        self.conv2 = nn.Conv2d(49, 10, kernel_size[1])

        # Fully connected layers, hidden unit of 32
        self.fc1 = nn.Linear(10*4*4, 100)
        self.fc2 = nn.Linear(100, 51) # 9 classifications

    def forward(self, img):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10*4*4)

        # use relu as activation function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x.shape)
        return x

## Training Code (not working yet)
# get the accuracy of the model
def get_accuracy(model, loader):
    correct = 0
    total = 0
    for features, labels in loader:
        # run on GPU if possible
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        outputs = model(features)
        # find the max predictive score
        pred = outputs.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += features.shape[0]
    return correct / total

def training(transfer_name = "alexnet", model = SimpleCNN(), bs = 27, ne = 1, lr = 0.001, custom_label = True):
    '''
    train the data
    transfer_name is "alexnet" or "vgg16"
    '''
    # use cross entropy loss for multi classification and adam optimizer
    if custom_label:
        criterion = nn.MultiLabelSoftMarginLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # load in data and create accuracy arrays
    train_loader, val_loader, test_loader = load_data(transfer_name, bs, False)
    train_loss, train_acc, val_acc, iters = [], [], [], []

    # Training
    start_time = time.time()
    i = 0
    print ("Training Started...")
    for epoch in range(ne):
        for features, labels in iter(train_loader):
            #print(labels.shape)
            #print(features.shape)
            #return
            # Run on GPU if possible
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            #print(features.shape)
            #print(labels.shape)
            #print(labels)
            #return 'f', 'g', 'l', 'i'
            optimizer.zero_grad()
            outputs = model(features)
            #print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        iters.append(i)
        i+=1
        train_loss.append(float(loss)/bs)           # compute loss
        train_acc.append(get_accuracy(model, train_loader)) # compute train_acc
        #val_err[epoch], val_loss[epoch] = evaluate_CNN(model, val_loader, criterion)
        val_acc.append(get_accuracy(model, val_loader))   # compute val_acc
        print("Epoch: " + str(epoch) + ', train acc: ' + str(train_acc[-1]) + ', train loss: ' + str(float(loss)) + ', valid acc: ' + str(val_acc[-1]))
        model_path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/{4}_models/model_{0}_bs{1}_lr{2}_epoch{3}".format(model.name, bs, lr, ne, transfer_name)
        torch.save(model.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed = (end_time - start_time)/60
    print("Total time elapsed in minutes: " + str(elapsed))
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    return iters, train_loss, train_acc, val_acc, model.name, bs, lr, ne, transfer_name

def plot_acc_loss(iters, losses, train_acc, val_acc, name, bs, lr, ne, transfer_name):
    path = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    ax1.plot(iters, np.dot(losses, bs), label = 'Train')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Training')
    #plt.setp(ax1.get_xticklabels(), rotation=45);

    ax2.plot(iters, train_acc, label="Train")
    ax2.plot(iters, val_acc, label="Validation")
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    plt.tight_layout()
    plt.legend()
    plt.savefig("{5}{4}_models/{0}_bs{1}_lr{2}_epoch{3}.png".format(name, bs, lr, ne, transfer_name, path))


# set up the model
use_cuda = True
model = SimpleCNN(kernel_size = [2,3], input = 512)
if use_cuda and torch.cuda.is_available():
    model.cuda()

## alexnet
iters, train_loss, train_acc, val_acc, name, bs, lr, ne, transfer_name = training('alexnet', model, 64, 15, 0.01, False)
plot_acc_loss(iter, train_loss, train_acc, val_acc, name, bs, lr, ne, transfer_name)

## vgg16 training
vgg_iters, vgg_train_loss, vgg_train_acc, vgg_val_acc, vgg_name, vgg_bs, vgg_lr, vgg_ne, vgg_transfer_name = training('vgg16', model, 64, 15, 0.01, False)
plot_acc_loss(vgg_iters, vgg_train_loss, vgg_train_acc, vgg_val_acc, vgg_name, vgg_bs, vgg_lr, vgg_ne, vgg_transfer_name)

## Sanity check (overfitting to transfer learning model to see if it works)









##### CODE BELOW DOESN'T WORK YET




## Functions
def evaluate_CNN(net, loader, criterion):
    """
    Taken from Lab 2
    """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    if torch.cuda.is_available():
        model.cuda()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        corr = (outputs > 0.0) != labels            #Check to see if this works
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss

def plot_CNN_training_curve(path):
    """
    Taken from Lab 2
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def train_CNN(model, train_set, val_set, batch = 16, lr = 0.001, num_epochs = 30):
  train_loader = dt.DataLoader(train_set, batch_size=batch, shuffle=True)
  val_loader = dt.DataLoader(val_set, batch_size=batch, shuffle=True)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  train_err = np.zeros(num_epochs)
  train_loss = np.zeros(num_epochs)
  val_err = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)
  if torch.cuda.is_available():
    model.cuda()
  start = time.time()
  for epoch in range(num_epochs):
    total_train_loss = 0.0
    total_train_err = 0.0
    total_epoch = 0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      if torch.cuda.is_available():
        inputs = inputs.cuda()
        labels = labels.cuda()
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      corr = (outputs > 0.0) != labels
      total_train_err += int(corr.sum())            #Check to see if this works
      total_train_loss += loss.item()
      total_epoch += len(labels)
    train_err[epoch] = float(total_train_err) / total_epoch
    train_loss[epoch] = float(total_train_loss) / (i+1)
    val_err[epoch], val_loss[epoch] = evaluate_CNN(model, val_loader, criterion)
    print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
    model_path = get_model_name(model.name, batch, lr, epoch)
    torch.save(model.state_dict(), model_path)
    present = time.time()
    elapsed = present - start
    print("Time elapsed: {:.2f} s".format(elapsed))
  print('Finished Training')
  end= time.time()
  elapsed_time = end - start
  print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
  epochs = np.arange(1, num_epochs + 1)
  np.savetxt("{}_train_err.csv".format(model_path), train_err)
  np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
  np.savetxt("{}_val_err.csv".format(model_path), val_err)
  np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

def CNN_wrapper(model, directory, win_len = 2048, hilbert = True, set_percent = 0.1, batch = 16, lr = 0.001, num_epochs = 30):
  train_set, val_set, test_set = get_CNN_data(directory, win_len, hilbert, set_percent)         #Get all datasets
  train_CNN(model, train_set, val_set, batch, lr, num_epochs)                                   #Train autoencoder
  path = get_model_name(model.name, batch, lr, num_epochs-1)                                    #Get path for plotting loss curves
  plot_CNN_training_curve(path)                                                                 #Plot loss curves
  test_loader = dt.DataLoader(train_set, batch_size=1, shuffle=True)                            #Get data loader for test data
  criterion = nn.BCEWithLogitsLoss()                                                            #Set criterion for test
  test_err, test_loss = evaluate_CNN(model, test_loader, criterion)                             #Get test accuracy and loss

