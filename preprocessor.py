import numpy as np
import librosa as lb

def draw_hilbert(dim, startx = 0, starty = 0, pos_step = True, xfirst = False):
    if dim == 2:
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
    else:
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

def gen_mel(file, sample_freq, n_fft, mel_bank):
  x, freq = lb.load(file, sample_freq)
  fourier = np.abs(lb.stft(x, n_fft))
  return np.matmul(mel_bank, fourier)

def plot_mels(mel_array, size):
  mels = mel_array.tolist()
  hilbert = draw_hilbert(size)
  if len(mels) != len(hilbert):
    raise ValueError('Number of mels does not match length of hilbert curve')
  array = np.zeros((size, size))
  for i in range(len(mels)):
    array[int(hilbert[i][0]), int(hilbert[i][1])] = mels[i]
  return array

def hilbert_data(file, sample_freq = 22050, n_fft = 2048, size = 128, fmin = 5, fmax = 8000):
  mel_bank = lb.filters.mel(sr = sample_freq, n_fft = n_fft, n_mels = size*size, fmin = fmin, fmax = fmax)
  mels = gen_mel(file, sample_freq, n_fft, mel_bank)
  array_length = mels.shape[1]
  array = plot_mels(mels[:, 0], size)
  arrays = np.expand_dims(array, 0)
  for i in range(1, array_length):
    arrays = np.append(arrays, np.expand_dims(plot_mels(mels[:, i], size), 0), axis = 0)
  return arrays
