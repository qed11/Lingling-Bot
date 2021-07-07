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

## import pretrained models
alexnet = torchvision.models.alexnet(pretrained=True)
vgg16 = torchvision.models.vgg.vgg16(pretrained=True)

##Set all the seeds to ensure reproducability
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

## Functions to save and create labelled dataset
def save_data(win_len = 2048, hilbert = True, save_name = None):
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
    labelled_dir = old_dir + "/Downloads/Audios/labelled/"
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
        print(k)

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
def load_data(name, bs):
    '''
    load the data from the files
    '''
    path = '/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert/features/'
    name = path + name
    # use ~60% of images including A-I for training (number 1-60)
    # use ~20% of images including A-I for validation (number 61-81)
    # use ~20% of images including A-I for testing (number 82-102/101)
    trainset = torchvision.datasets.DatasetFolder(name + '_train_duplicates', loader = torch.load, extensions = ('.tensor'))
    valset = torchvision.datasets.DatasetFolder(name + '_val', loader = torch.load, extensions = ('.tensor'))
    testset = torchvision.datasets.DatasetFolder(name + '_test', loader = torch.load, extensions = ('.tensor'))

    # load data in and return
    trainload = dt.DataLoader(trainset, batch_size = bs, shuffle = True)
    valload = dt.DataLoader(valset, batch_size = bs, shuffle = True)
    testload = dt.DataLoader(testset, batch_size = bs, shuffle = True)
    return trainload, valload, testload
## load the loaders for alexnet and vgg16
train_alex, val_alex, test_alex = load_data("alexnet", 1)
train_vgg, val_vgg, test_vgg = load_data('vgg16', 1)

## Create Simple Model CNN for Alexnet and vgg16
class SimpleCNN(nn.Module):
    def __init__(self, kernel_size = [2,2]):
        super(AlexNew, self).__init__()
        self.name = "SimpleCNN"

        # 2 convolution layers
        self.conv1 = nn.Conv2d(256, 49, kernel_size[0])
        self.conv2 = nn.Conv2d(49, 10, kernel_size[1])

        # Fully connected layers, hidden unit of 32
        self.fc1 = nn.Linear(10*4*4, 32)
        self.fc2 = nn.Linear(32, 9) # 9 classifications

    def forward(self, img):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 10*4*4)

        # use relu as activation function
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## Training Code


##
for i in range(len(label.squeeze())):
    if int(label.squeeze()[i].item()) == 1:
        name = str(name) + str(label_dic[i]) + "_"
##
i = 0
for image, label in training_hilb:
    print(image)
    print(image.shape)
    print(label)
    print(label.shape)

    i+= 1
    img = np.transpose(image, [1,2,0])
    plt.imshow(img)
    #plt.show()
    if i > 2:
        break

len(training_hilb)







##### CODE BELOW DOESN'T WORK YET
## Save Features

##

##

## Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as dt
import torchvision.transforms as transforms
import time
import os
import random
from PIL import Image
import librosa as lb
os.chdir("/Users/sarinaxi/Desktop/Lingling-Bot")
from preprocessor import autoencoder_hilbert_data, autoencoder_spectrogram_data

##Set all the seeds to ensure reproducability
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

## Functions
def sound2image(path, savepath, win_len = 2048, hilbert = True):
    """
    path is string to where the sound files are
    win_len is int for the window size
    hilbert is booleam whether use hilbert to path image or not
    """

    n = 1
    for file in os.listdir(path):
        # For each .wav file in the downloaded path
        if file.endswith(".wav"):
            # if want hilbert
            file = '/Users/sarinaxi/Desktop/Lingling-Bot/Downloads/Audios/' + file
            if hilbert:
                start_time = time.time()
                data = autoencoder_hilbert_data(file, win_len = win_len)
                end_time = time.time()
                diff = end_time - start_time
                print("Hilbert conversion complete: " + str(diff) + " seconds.")
                for array in data:
                    #plt.imshow(array)
                    im = Image.fromarray(array.astype('uint8'), 'RGB')
                    im.show()
                    return
                    plt.imshow(im)
                    im.save(savepath + str(n) + ".jpeg") #newpath is where to save the image data)

                    n += 1
                    return
            # else if don't want hilbert
            else:
                data = autoencoder_spectrogram_data(file, win_len = win_len)
                for array in data:
                    im = Image.fromarray(array)
                    im.save(savepath + str(n) + ".png") #newpath to replace)
                    n += 1

def load_data(newpath, set_percent = 0.1):
    """
    load the data from newpath which contains image folders
    set_percent is the % of data in validation/testing
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5/3))])   #Make sure images are tensors, lie within 0 and 1
    dataset = torchvision.datasets.ImageFolder(newpath, transform = transform) #newpath to replace)  #Turn the images into a dataset
    set_size = int(set_percent*len(dataset))
    #Return datasets in order of training set, validation set, test set
    return dt.random_split(dataset, [len(dataset) - 2*set_size, set_size, set_size])

sound2image(path = "/Users/sarinaxi/Desktop/Lingling-Bot/Downloads/Audios/", savepath = "/Users/sarinaxi/Desktop/Lingling-Bot/Data/Hilbert")

