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
os.chdir("/Users/sarinaxi/Desktop/Lingling-Bot")
from youtube_scraper import download_csv_audio
#from preprocessor import autoencoder_hilbert_data, autoencoder_spectrogram_data

##Set all the seeds to ensure reproducability
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

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
def get_autoencoder_data(from_csv= False, csv_path = None, win_len = 2048, hilbert = True, set_percent = 0.1):
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
        for file in os.listdir(folder):
            # check if file ends with wav
            if file.endswith('.wav'):
                # If hilbert is required, save the .wav files as hilbert curve data
                if hilbert:
                    new_path = old_dir + "/Data/Hilbert/" + folder + '/'
                    if os.path.isdir(new_path) is False:
                        os.mkdir(new_path)
                    file_path = os.path.abspath(folder + "/" + file)
                    print("file" + file)
                    print("file_path" + file_path)
                    data = autoencoder_hilbert_data(file_path, win_len = win_len)
                    for array in data:
                        im = Image.fromarray(array)
                        if im.mode == "F":
                            im = im.convert('RGB') # convert to RGB because cannot save to float
                        im.save(new_path + str(n) + ".png") #newpath is where to save the image data)
                        n += 1
                else: #Else, save the .wav files as spectrogram data
                    new_path = old_dir + "/Data/Spectrogram/" + folder + "/"
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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5/3))])   #Make sure images are tensors, lie within 0 and 1
    dataset = torchvision.datasets.ImageFolder(new_path, transform = transform) #newpath to replace) #Turn the images into a dataset
    set_size = int(set_percent*len(dataset))                                                        #Length of test and validation sets
    return dt.random_split(dataset, [len(dataset) - 2*set_size, set_size, set_size])                #Return datasets in order of training set, validation set, test set

## Test the Functions
get_autoencoder_data()
