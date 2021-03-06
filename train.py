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
#from youtube_scrapper import download_csv_audio
#from preprocessor import autoencoder_hilbert_data, autoencoder_spectrogram_data

#Set all the seeds to ensure reproducability
random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)

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

def get_autoencoder_data(csv_path, win_len = 2048, hilbert = True, set_percent = 0.1):
  download_csv_audio(csv_path)                                                                    #Download all the youtube links as .wav files
  n = 1
  for file in os.listdir(directory #where the youtube files were saved):
    if file.endswith(".wav"):                                                                     #For each .wav file in the downloaded path
      if hilbert:                                                                                 #If hilbert is required, save the .wav files as hilbert curve data
        data = autoencoder_hilbert_data(file, win_len = win_len)
        for array in data:
          norm = (array - np.min(array))/(np.max(array) - np.min(array))*255
          norm = norm.astype(np.uint8)
          im = Image.fromarray(norm)
          im.save(newpath + str(n) + ".png" #newpath is where to save the image data)
      else:                                                                                       #Else, save the .wav files as spectrogram data
        data = autoencoder_spectrogram_data(file, win_len = win_len)
        for array in data:
          norm = (array - np.min(array))/(np.max(array) - np.min(array))*255
          norm = norm.astype(np.uint8)
          im = Image.fromarray(norm)
          im.save(newpath + str(n) + ".png" #newpath to replace)
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5/3))])   #Make sure images are tensors, lie within 0 and 1
  dataset = torchvision.datasets.ImageFolder(newpath, transform = transform #newpath to replace)  #Turn the images into a dataset
  set_size = int(set_percent*len(dataset))                                                        #Length of test and validation sets
  return dt.random_split(dataset, [len(dataset) - 2*set_size, set_size, set_size])                #Return datasets in order of training set, validation set, test set

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
