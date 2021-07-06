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
from preprocessor import autoencoder_hilbert_data, autoencoder_spectrogram_data

## import pretrained models
alexnet = torchvision.models.alexnet(pretrained=True)
inception = torchvision.models.inception.inception_v3(pretrained=True)
vgg16 = torchvision.models.vgg.vgg16(pretrained=True)
resnet18 = torchvision.models.resnet.resnet18(pretrained=True)

## Save Features
cur_path = os.getcwd()
data_path = cur_path + "/Images/"

bs = 1
train_loader, val_loader, test_loader = load_data()
##
# save features to folder as tensors, path as the path to save
def save_features(loader, path, model):
    i = 0
    for images, label in loader:
        features = model.features(images)
        features = torch.from_numpy(features.detach().numpy())
        new_folder = path + str(types[label])
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)
        torch.save(features.squeeze(0), new_folder + "/" + str(i) + '.tensor')
        i += 1

save_features(train_loader, cur_path + "/Features/Train/Alex", alexnet)
save_features(val_loader, cur_path + "/Features/Val/Alex", alexnet)
save_features(test_loader, cur_path + "/Features/Test/Alex", alexnet)

save_features(train_loader, cur_path + "/Features/Train/Incep", inception)
save_features(val_loader, cur_path + "/Features/Val/Incep", inception)
save_features(test_loader, cur_path + "/Features/Test/Incep", inception)

save_features(train_loader, cur_path + "/Features/Train/Vgg16", vgg16)
save_features(val_loader, cur_path + "/Features/Val/Vgg16", vgg16)
save_features(test_loader, cur_path + "/Features/Test/Vgg16", vgg16)

save_features(train_loader, cur_path + "/Features/Train/resnet18", resnet18)
save_features(val_loader, cur_path + "/Features/Val/resnet18", resnet18)
save_features(test_loader, cur_path + "/Features/Test/resnet18", resnet18)

##

def load_data_alex(path, bs):
    # specify transformations applied to the images
    transformations = transforms.Compose([
        transforms.CenterCrop(224), # make sure the size is right
        transforms.ToTensor(),
    ])

    trainset = datasets.DatasetFolder(path + 'train', loader = torch.load, extensions = ('.tensor'))
    valset = datasets.DatasetFolder(path + 'val', loader = torch.load, extensions = ('.tensor'))
    testset = datasets.DatasetFolder(path + 'test', loader = torch.load, extensions = ('.tensor'))

    # load data in and return
    trainload = torch.utils.data.DataLoader(trainset, batch_size = bs, shuffle = True)
    valload = torch.utils.data.DataLoader(valset, batch_size = bs, shuffle = True)
    testload = torch.utils.data.DataLoader(testset, batch_size = bs, shuffle = True)
    return trainload, valload, testload



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

## Kevin Functions
def get_model_name(name, batch_size, learning_rate, epoch):
    """
    Taken from Lab 2
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def evaluate_autoencoder(net, loader, criterion):
    """
    Parts taken from Lab 2
    """
    total_loss = 0.0
    if torch.cuda.is_available():
        model.cuda()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
          inputs = inputs.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        total_loss += loss.item()
    loss = float(total_loss) / (i + 1)
    return loss

def plot_autoencoder_training_curve(path):
    """
    Parts taken from Lab 2
    """
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()


def train_autoencoder(model, train_set, val_set, nf = 0.4, batch = 16, lr = 0.001, num_epochs = 30):
  """
  Parts taken from Lab 2
  """
  train_loader = dt.DataLoader(train_set, batch_size=batch, shuffle=True)
  val_loader = dt.DataLoader(val_set, batch_size=batch, shuffle=True)
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  train_loss = np.zeros(num_epochs)
  val_loss = np.zeros(num_epochs)
  if torch.cuda.is_available():
    model.cuda()
  start = time.time()
  for epoch in range(num_epochs):
    total_train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      noisy_inputs = inputs + nf * torch.randn(*inputs.shape)
      noisy_inputs = np.clip(noisy_inputs, 0., 1.)
      if torch.cuda.is_available():
        inputs = inputs.cuda()
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, inputs)
      loss.backward()
      optimizer.step()
      total_train_loss += loss.item()
    train_loss[epoch] = float(total_train_loss) / (i+1)
    val_loss[epoch] = evaluate_autoencoder(model, val_loader, criterion)
    print(("Epoch {}: Train loss: {} , Validation loss: {}").format(
                   epoch + 1,
                   train_loss[epoch],
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
  np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
  np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

def autoencoder_wrapper(model, csv_path, win_len = 2048, hilbert = True, set_percent = 0.1, nf = 0.4, batch = 16, lr = 0.001, num_epochs = 30):
  """
  Wraps all autoencoder training stuff together so you just have to run one function, sit back, and relax.
  """
  train_set, val_set, test_set = get_autoencoder_data(csv_path, win_len, hilbert, set_percent)  #Get all datasets
  train_autoencoder(model, train_set, val_set, nf, batch, lr, num_epochs)                       #Train autoencoder
  path = get_model_name(model.name, batch, lr, num_epochs-1)                                    #Get path for plotting loss curves
  plot_autoencoder_training_curve(path)                                                                     #Plot loss curves
  test_loader = dt.DataLoader(train_set, batch_size=1, shuffle=True)                            #Get data loader for test data
  tests = np.random.rand(1, 20)                                                                 #Randomly choose 20 test images
  ls = []
  for num in tests:
    ls.append(int(num*(len(test_loader)-1)))                                                    #Turn the random numbers between 0 and 1 to callable indices
  j = 0
  k = 0
  test_data = []
  for i, data in enumerate(test_loader, 0):                                                     #Show original pictures (based on code from Lab 2)
    if not i in ls:
      continue
    images, labels = data
    test_data.append(images[0])
    plt.subplot(4, 5, j+1)
    plt.axis('off')
    plot.imshow(images[0])
    j += 1
  plt.show()
  for image in test_data:                                                                       #Show encoded and decoded pictures
    output = model(image)
    plt.subplot(4, 5, k+1)
    plt.axis('off')
    plot.imshow(output)
    k += 1
  plt.show()

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
        #corr = outputs.max(1).indices != labels                Change - for later patch
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


def get_CNN_data(directory, win_len = 2048, hilbert = True, set_percent = 0.1):
  dataset = None
  labels = None
  for file in os.listdir(directory):
    if file.endswith(".wav"):                                                                     #For each .wav file in the downloaded path
      if hilbert:                                                                                 #If hilbert is required, save the .wav files as hilbert curve data
        data, label = hilbert_data(file, win_len = win_len)
        data, label = torch.Tensor(list(data)), torch.Tensor(list(label))                         #Convert data and labels to tensors
        if dataset == None:
          dataset = data
          labels = label
        else:
          dataset = torch.cat(dataset, data)
          labels = torch.cat(labels, label)
      else:                                                                                       #Else, save the .wav files as spectrogram data
        data, label = spectrogram_data(file, win_len = win_len)
        data, label = torch.Tensor(list(data)), torch.Tensor(list(label))                         #Convert data and labels to tensors
        if dataset == None:
          dataset = data
          labels = label
        else:
          dataset = torch.cat(dataset, data)
          labels = torch.cat(labels, label)
  full_set = dt.TensorDataset(dataset, labels)
  return dt.random_split(full_set, [len(dataset) - 2*set_size, set_size, set_size])                #Return datasets in order of training set, validation set, test set












## Functions
def autoencoder_hilbert_data(file, sample_freq = 22050, n_fft = 65536, win_len = 2048, size = 128, fmin = 5, fmax = 8000):
    """
    Given a file, returns an array of images made by mapping the intensities of the mel spectrums using hilbert curves at points in time, without instrument labels.
    """
    mel_bank = lb.filters.mel(sr = sample_freq, n_fft = n_fft, n_mels = size*size, fmin = fmin, fmax = fmax)  #Get mel transformation matrix
    mels = gen_mel(file, sample_freq, n_fft, mel_bank, win_len)                                               #Get the value of the mel bins for each point in time
    array_length = mels.shape[1]                                                                              #Get length of the mels array
    array = np.expand_dims(plot_mels(mels[:, 0], size), 0)                                                    #Initialize the list of mapped mel spectrums
    for i in range(1, array_length):
        array = np.append(array, np.expand_dims(plot_mels(mels[:, i], size), 0), axis = 0)                    #Keep adding on mapped mel spectrums
    return array

def autoencoder_spectrogram_data(file, sample_freq = 22050, n_ftt = 65536, win_len = 2048):
    x, freq = lb.load(file, sample_freq)
    data = lb.amplitude_to_db(np.abs(lb.feature.melspectrogram(x, freq, n_ftt = n_ftt, win_length = win_len)), ref = np.max)  #Get spectrogram
    size = len(data)                                                                                                          #Get size of spectrogram (num bins)
    add_on = size - 1
    num_ints = floor(len(data[0])/size)*4                                                                                     #Get number of data points
    arrays = np.expand_dims(data[:, 0:127], 0)                                                                                #Initialize the list of spectrograms
    for i in range(1, num_ints):
        stt_ind = i*size/4
        fin_ind = i*size/4 + add_on
        arrays = np.append(arrays, np.expand_dims(data[:, stt_ind:fin_ind], 0), axis = 0)                                       #Keep adding on spectrograms
    return arrays

def save_into_images(from_csv= False, csv_path = None, win_len = 2048, hilbert = True, set_percent = 0.1):
    """
    Either get from a csv file with youtube links using from_csv and csv_path
    or directly from a folder with all the labelled clips
    Option of using a hilbert path or not and changing the size of the dataset through win_length
    set_percentage sets the % of data in test and validation
    """
    if from_csv == True and csv_path is not None:
        download_csv_audio(csv_path)

    n = 1
    # old_dir assumes you are in the current file directory
    old_dir = os.getcwd()
    os.chdir(old_dir + "/Downloads/Audios/")
    new_dir = os.getcwd()
    for folder in os.listdir(new_dir): #where the youtube files were saved):
        # go into each folder piece
        if os.path.isdir(folder) is True:
            for file in os.listdir(folder):
                # check if file ends with wav
                if file.endswith('.wav'):
                    # If hilbert is required, save the .wav files as hilbert curve data
                    if hilbert:
                        new_path = old_dir + "/Data/Hilbert/"
                        if os.path.isdir(new_path) is False:
                            os.mkdir(new_path)
                        file_path = os.path.abspath(folder + "/" + file)
                        print("file" + file)
                        print("file_path" + file_path)
                        data = autoencoder_hilbert_data(file_path, win_len = win_len)
                        for array in data:
                            im = Image.fromarray(array)
                            im.save(new_path + str(n) + ".png") #newpath is where to save the image data)
                            n += 1
                    else: #Else, save the .wav files as spectrogram data
                        new_path = old_dir + "/Data/Spectrogram/"
                        if os.path.isdir(new_path) is False:
                            os.mkdir(new_path)
                        file_path = os.path.abspath(folder + "/" + file)
                        print(file_path)
                        data = autoencoder_spectrogram_data(file_path, win_len = win_len)
                        for array in data:
                            im = Image.fromarray(array)
                            if im.mode == "F":
                                im = im.convert('RGB') # convert to RGB because cannot save to float
                            im.save(new_path + str(n) + ".png") #newpath to replace)
                            n += 1
def load_data(new_path):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5/3))])   #Make sure images are tensors, lie within 0 and 1
    dataset = torchvision.datasets.ImageFolder(new_path, transform = transform) #newpath to replace) #Turn the images into a dataset
    set_size = int(set_percent*len(dataset))                                                        #Length of test and validation sets
    return dt.random_split(dataset, [len(dataset) - 2*set_size, set_size, set_size])        #Return datasets in order of training set, validation set, test set

## Test the Functions
#load_data("/Data/Hilbert/" + folder + '/')
