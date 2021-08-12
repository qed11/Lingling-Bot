import numpy as np
import librosa as lb
from math import floor

def draw_hilbert(dim, startx = 0, starty = 0, pos_step = True, xfirst = False):
    """
    Given a dimension that is a power of 2, return a list of points that forms a Hilbert Curve when connected one-by-one.
    """
    if dim == 2:                                                                                                #If the dimension is 2, draw a u based on arguments pos_step and xfirst
        if pos_step:
            if not xfirst:
                return [(startx, starty), (startx, starty+1), (startx+1, starty+1), (startx+1, starty)]
            else:
                return [(startx, starty), (startx+1, starty), (startx+1, starty+1), (startx, starty+1)]
        else:
            if not xfirst:
                return [(startx, starty), (startx, starty-1), (startx-1, starty-1), (startx-1, starty)]
            else:
                return [(startx, starty), (startx-1, starty), (startx-1, starty-1), (startx, starty-1)]
    else:                                                                                                       #If the dimension is not 2, compute the starting points of the 4 smaller hilbert curves
        sign = 1
        if not pos_step:
            sign = -1
        if not xfirst:
            startx2, starty2 = startx, starty+dim*sign/2
            startx3, starty3 = startx2+dim*sign/2, starty2
            startx4, starty4 = startx3+(dim/2-1)*sign, starty3-sign
        else:
            startx2, starty2 = startx+dim*sign/2, starty
            startx3, starty3 = startx2, starty2+dim*sign/2
            startx4, starty4 = startx3-sign, starty3+(dim/2-1)*sign
        return draw_hilbert(dim/2, startx, starty, pos_step, not xfirst) + draw_hilbert(dim/2, startx2, starty2, pos_step, xfirst) + draw_hilbert(dim/2, startx3, starty3, pos_step, xfirst) + draw_hilbert(dim/2, startx4, starty4, not pos_step, not xfirst)

def gen_mel(file, sample_freq, n_fft, mel_bank, win_len):
  """
  Given a file and a mel transformation matrix, returns the mel spectrums across time in that file.
  """
  x, freq = lb.load(file, sample_freq)                                                          #Load in the file
  fourier = np.abs(lb.stft(x, n_fft, win_length = int(win_len)))                                     #Take the magnitudes of a short time fourier transform
  return lb.amplitude_to_db(np.matmul(mel_bank, fourier), ref = np.max)                         #Multiply the fourier transform bins with the mel transformation matrix to get mel-scaled bins

def plot_mels(mel_array, size):
  """
  Given a 1D np array representing the mel spectrum of a point in time and the size of the appropriate hilbert curve, return the mel spectrum mapped to a 2D surface using that hilbert curve as a 2D array.
  """
  mels = mel_array.tolist()                                                     #Change numpy array to list for easier indexing
  hilbert = draw_hilbert(size)                                                  #Get the positions of the mel bins to map the mels onto a 2D array
  if len(mels) != len(hilbert):
    raise ValueError('Number of mels does not match length of hilbert curve')
  array = np.zeros((size, size))                                                #Initialize array, set to 0s
  for i in range(len(mels)):
    array[int(hilbert[i][0]), int(hilbert[i][1])] = mels[i]                     #Fill array
  return array


label_dic = ['VLN', 'VLA', 'CEL', 'DBS', 'FLT', 'CLT', 'OBO', 'BSN', 'TPT', 'FHN', 'TBN', 'TUB', 'PNO', 'PER']

def label_data(filename):
  fn = filename.upper()
  labels = fn.split('.')[0].split('_')
  #print(labels)
  out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  for i, instrument in enumerate(label_dic):
    for j in labels:
      if instrument in j:
        out[i] = 1
  return out

def hilbert_data(file, sample_freq = 22050, n_fft = 65536, win_len = 2048, size = 128, fmin = 5, fmax = 8000):
  """
  Given a file, returns an array of images made by mapping the intensities of the mel spectrums using hilbert curves at points in time, with instrument labels.
  """
  label = np.array(label_data(file))
  mel_bank = lb.filters.mel(sr = sample_freq, n_fft = n_fft, n_mels = size*size, fmin = fmin, fmax = fmax)  #Get mel transformation matrix
  mels = gen_mel(file, sample_freq, n_fft, mel_bank, win_len)                                               #Get the value of the mel bins for each point in time
  array_length = mels.shape[1]                                                                              #Get length of the mels array
  arrays = np.expand_dims(plot_mels(mels[:, 0], size), 0)                                                   #Initialize the list of mapped mel spectrums
  label_array = np.expand_dims(label, 0)                                                                    #Initialize array of labels
  for i in range(1, array_length):
    arrays = np.append(arrays, np.expand_dims(plot_mels(mels[:, i], size), 0), axis = 0)                    #Keep adding on mapped mel spectrums
    label_array = np.append(label_array, np.expand_dims(label, 0), axis = 0)                                #Keep adding on labels
  return arrays, label_array

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

def spectrogram_data(file, sample_freq = 22050, n_fft = 65536, win_len = 2048):
  label = np.array(label_data(file))
  x, freq = lb.load(file, sample_freq)
  data = lb.amplitude_to_db(np.abs(lb.feature.melspectrogram(x, freq, n_fft = n_fft, win_length = win_len)), ref = np.max)  #Get spectrogram
  size = len(data)                                                                                                          #Get size of spectrogram (num bins)
  num_ints = floor(len(data[0])/size)*4 - 3                                                                             #Get number of data points
  arrays = np.expand_dims(data[:, 0:size], 0)                                                                                #Initialize the list of spectrograms
  label_array = np.expand_dims(label, 0)                                                                                    #Initialize array of labels
  for i in range(1, num_ints):
    stt_ind = int(i*size/4)
    fin_ind = int(i*size/4 + size)
    arrays = np.append(arrays, np.expand_dims(data[:, stt_ind:fin_ind], 0), axis = 0)                                       #Keep adding on spectrograms
    label_array = np.append(label_array, np.expand_dims(label, 0), axis = 0)                                                #Keep adding on labels
  return arrays, label_array

def autoencoder_spectrogram_data(file, sample_freq = 22050, n_fft = 65536, win_len = 2048):
  x, freq = lb.load(file, sample_freq)
  data = lb.amplitude_to_db(np.abs(lb.feature.melspectrogram(x, freq, n_fft = n_fft, win_length = win_len)), ref = np.max)  #Get spectrogram
  size = len(data)                                                                                                          #Get size of spectrogram (num bins)
  num_ints = floor(len(data[0])/size)*4 - 3                                                                                     #Get number of data points
  arrays = np.expand_dims(data[:, 0:size], 0)                                                                                #Initialize the list of spectrograms
  for i in range(1, num_ints):
    stt_ind = int(i*size/4)
    fin_ind = int(i*size/4 + size)
    arrays = np.append(arrays, np.expand_dims(data[:, stt_ind:fin_ind], 0), axis = 0)                                       #Keep adding on spectrogram
  return arrays
