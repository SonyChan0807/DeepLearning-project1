import os
import random
import numpy as np
import matplotlib.pyplot as plt

import dlc_bci as bci

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.signal import hilbert, chirp
from scipy import signal

def convert_to_one_hot_labels(input, target):
    """Function to convert 2 class categorical labels to one-hot labels
    Args:
        input       (torch.FloatTensor) : Tensor with dimensions as (number of samples x rows x columns)
        target      (torch.FloatTensor) : Tensor with dimensions as (number of samples x 1)
        
    Returns:
        tmp         (torch.FloatTensor) : Tensor with dimensions as (number of samples x number of classes)  
    """
    
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    for k in range(0, target.size(0)):
        tmp[k, target[k]] = 1
    return tmp

def replicate_samples(tr_input, tr_target, no_samples):   
    """Function to replicate data through random sampling
    Args:
        tr_input       (torch.FloatTensor) : Tensor with dimensions as (number of samples x rows x columns)
        tr_target      (torch.FloatTensor) : Tensor with dimensions as (number of samples x 1)
        no_samples     (int)               : number of additional samples to generate
        
    Returns:
        tr_input       (torch.FloatTensor) : Tensor with dimensions as ([number of samples + nb_samples] x rows x columns)
        tr_input       (torch.FloatTensor) : Tensor with dimensions as ([number of samples + nb_samples] x 1)    
    """     
    
    ind       = np.random.choice(np.arange(0,np.shape(tr_input)[0]), no_samples)[None]
    tr_input  = torch.cat((tr_input, tr_input[ind]))
    tr_target = torch.cat((tr_target, tr_target[ind]))
    
    return tr_input, tr_target

def cross_validation_batch(size, k):
    """Function to generate indices for k-fold cross validation
    Args:
        size           (int)              : input size
        k              (int)              : k-fold
    """    

    num = size // k
    rand_index = np.arange(size)
    np.random.shuffle(rand_index)
    
    tr_indices = []
    val_indices = []
    for i in range(k):
        
        if i != k-1:
            val_ind = rand_index[i*num:(i+1)*num]
        else:
            val_ind = rand_index[i*num:]
        
        val_indices.append(torch.from_numpy(val_ind))
        tr_indices.append(torch.from_numpy(np.array(list(set(range(size)) - set(val_ind)))))
    
    return tr_indices, val_indices

def conv_lowpass(signal, N):    
    """Function to generate signals after low-pass filtering
    Args:
        signal         (torch.FloatTensor): Matrix with size N x C x L
        N              (int)              : size of moving average filter
    """    
    return np.convolve(signal, np.ones((N,))/N, mode='valid')

def lowpass_filtering(signal, N):
    """Function to generate signals after low-pass filtering
    Args:
        signal         (torch.FloatTensor): Matrix with size N x C x L
        N              (int)              : size of moving average filter
    """    

    sample = conv_lowpass(signal[0,0,:].numpy(), N)
    
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(conv_lowpass(channel.numpy(), N))
    
    return signal_filtered    

def downsampling_filtering(signal, Nd):
    """Function to generate signals after downsampling
    Args:
        signal         (torch.FloatTensor): Matrix with size N x C x L
        Nd             (int)              : Downsampling size
    """    

    sample = signal[0,0,:].numpy()[::Nd]
    
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = channel[::Nd]
    
    return signal_filtered

def dc_blocker(signal, alpha = 0.9):
    """Function to generate signals after dc blocking
    Args:
        signal         (torch.FloatTensor): Matrix with size N x C x L
        alpha          (float)              : parameters for dc blocking 
    """    

    y = np.zeros(signal.shape[0]+1)
    y[0] = signal[0]
    
    for i in range(1, len(signal)):
        y[i] = alpha * y[i-1] + signal[i] - signal[i-1]
        i+=1
    
    return y

def dc_blocker_filtering(signal, alpha_ = 0.9):
    """Function to generate signals after dc blocking
    Args:
        signal         (torch.FloatTensor): Matrix with size N x C x L
        alpha          (float)              : parameters for dc blocking 
    """    

    sample = dc_blocker(signal[0,0,:].numpy(), alpha=alpha_)
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])
    
    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(dc_blocker(channel.numpy(), alpha=alpha_))
    
    return signal_filtered


def fft_data(signal, size, with_phase):
    """Function to generate spectrum of signals (without DC value)
    Args:
        signal         (torch.FloatTensor): Matrix with size N x C x L
        size           (int)              : number of frequency components we want
        with_phase     (boolean)          : include the data of phase or not 
    """    

    sample_fft = np.fft.fft(signal)
    sample_angle = np.angle(sample_fft)
    
    if with_phase:
        return np.append(abs(sample_fft)[1:size+1],sample_angle[1:size+1])
    else:
        return abs(sample_fft)[1:size+1]

def fft_input_generation(signal, size, with_phase):
    """Function to generate spectrum of signals (without DC value)
    Args:
        signal         (torch.FloatTensor): Matrix with size N x C x L
        size           (int)              : number of frequency components we want
        with_phase     (boolean)          : include the data of phase or not 
    """    

    sample = fft_data(signal[0,0,:].numpy(), size, with_phase)
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])
    
    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(fft_data(channel.numpy(), size, with_phase))
    return signal_filtered

def spectrogram_data(signal_, freq, npers):
    """Function to generate the data of spectrogram
    Args:
        signal_        (torch.FloatTensor): Matrix with size N x C x L
        freq           (int)              : frequency
        nperseg        (int)          : number of data per segment
    """    

    f, t, Sxx = signal.spectrogram(signal_, freq, nperseg=npers)
    return Sxx.ravel()

def spectrogram_input_generation(signal_, freq = 100, npers = 10):
    """Function to generate the data of spectrogram
    Args:
        signal_        (torch.FloatTensor): Matrix with size N x C x L
        freq           (int)              : frequency
        nperseg        (int)          : number of data per segment
    """    

    sample = spectrogram_data(signal_[0,0,:], freq, npers)
    signal_filtered = torch.zeros([signal_.shape[0], signal_.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal_):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(spectrogram_data(channel, freq, npers))
    return signal_filtered
    

def peak_detector(signal, N):
    """Function to generate the data of spectrogram
    Args:
        signal        (torch.FloatTensor): Matrix with size N x C x L
        N             (int)              : 
    """    

    length = signal.shape[0]
    result = np.zeros(length - N)
    for i in range(length - N):
        result[i] = np.max(signal[i:i+N])
    
    return result
def peak_detector_filtering(signal, N):
    """Function to fetch the local max in time series
    Args:
        signal        (torch.FloatTensor): Matrix with size N x C x L
        N             (int)              : 
    """    

    sample = peak_detector(signal[0,0,:].numpy(), N)
    
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(peak_detector(channel.numpy(), N))
    
    return signal_filtered    

def channel_used(signal, reserve_list):
    """Function to choose the channels to be used
    Args:
        signal        (torch.FloatTensor): Matrix with size N x C x L
        reserve_list  (list)             : list to indicate which channel to be reserved 
    """    

    signal_filtered = torch.zeros([signal.shape[0], len(reserve_list), signal.shape[2]])
    
    for idx, ch in enumerate(reserve_list):
        signal_filtered[:,idx,:] = signal[:,ch,:]
    
    return signal_filtered

def augment(signal, target, increase_size):
    """Function to augment the data
    Args:
        input         (torch.FloatTensor) : Tensor with dimensions as (number of samples x rows x columns)
        target        (torch.FloatTensor) : Tensor with dimensions as (number of samples x 1)
        increase_size (int)               : the number of data to be increased
    """

    tmp = torch.zeros(increase_size, signal.size(1), signal.size(2))
    aug_target = torch.zeros(increase_size)
    ind_max = signal.size(0)
    
    ls1 = np.where(target.data.numpy()==1)[0]
    ls2 = np.where(target.data.numpy()==0)[0]
    
    m = int(increase_size/2)
    aug_target[0:m] = 1
    aug_target[m:] = 0

    for i in range(m):
        a = signal[random.choice(ls1),:,:]
        b = signal[random.choice(ls1),:,:]
        tmp[i,:,:] = (a+b)/2
    
    for i in range(m, increase_size):
        a = signal[random.choice(ls2),:,:]
        b = signal[random.choice(ls2),:,:]
        tmp[i,:,:] = (a+b)/2
    
    return torch.cat((signal, tmp), 0), torch.cat((target, Variable(aug_target.long())), 0)
