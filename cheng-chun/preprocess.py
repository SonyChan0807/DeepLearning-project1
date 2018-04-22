import os
import random
import numpy as np
import matplotlib.pyplot as plt

import dlc_bci as bci
from dlc_practical_prologue import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.signal import hilbert, chirp
from scipy import signal

def conv_lowpass(signal, N):    
    return np.convolve(signal, np.ones((N,))/N, mode='valid')

def lowpass_filtering(signal, N):
    sample = conv_lowpass(signal[0,0,:].numpy(), N)
    
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(conv_lowpass(channel.numpy(), N))
    
    return signal_filtered    

def downsampling_filtering(signal, Nd):
    sample = signal[0,0,:].numpy()[::Nd]
    
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
#             signal_filtered[idx1, idx2, :] = torch.from_numpy(channel.numpy()[::Nd])
            signal_filtered[idx1, idx2, :] = channel[::Nd]
    
    return signal_filtered

def dc_blocker(signal, alpha = 0.9):
    y = np.zeros(signal.shape[0]+1)
    y[0] = signal[0]
    
    for i in range(1, len(signal)):
        y[i] = alpha * y[i-1] + signal[i] - signal[i-1]
        i+=1
    
    return y

def dc_blocker_filtering(signal, alpha_ = 0.9):
    sample = dc_blocker(signal[0,0,:].numpy(), alpha=alpha_)
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])
    
    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(dc_blocker(channel.numpy(), alpha=alpha_))
    
    return signal_filtered


def fft_data(signal, size, with_phase):
    sample_fft = np.fft.fft(signal)
    sample_angle = np.angle(sample_fft)
    
    if with_phase:
        return np.append(abs(sample_fft)[1:size+1],sample_angle[1:size+1])
    else:
        return abs(sample_fft)[1:size+1]

def fft_input_generation(signal, size, with_phase):
    sample = fft_data(signal[0,0,:].numpy(), size, with_phase)
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])
    
    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(fft_data(channel.numpy(), size, with_phase))
    return signal_filtered

def spectrogram_data(signal_, freq, npers):
    f, t, Sxx = signal.spectrogram(signal_, freq, nperseg=npers)
    return Sxx.ravel()

def spectrogram_input_generation(signal_, freq = 100, npers = 10):
    sample = spectrogram_data(signal_[0,0,:], freq, npers)
    signal_filtered = torch.zeros([signal_.shape[0], signal_.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal_):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(spectrogram_data(channel, freq, npers))
    return signal_filtered
    
def peak_detector(signal, N):
    length = signal.shape[0]
    result = np.zeros(length - N)
    for i in range(length - N):
        result[i] = np.max(signal[i:i+N])
    
    return result
def peak_detector_filtering(signal, N):
    sample = peak_detector(signal[0,0,:].numpy(), N)
    
#     print(signal[0,0,:].shape[0])
    signal_filtered = torch.zeros([signal.shape[0], signal.shape[1], sample.shape[0]])

    for idx1, data in enumerate(signal):
        for idx2, channel in enumerate(data):
            signal_filtered[idx1, idx2, :] = torch.from_numpy(peak_detector(channel.numpy(), N))
    
    return signal_filtered    

def channel_used(signal, reserve_list):
    signal_filtered = torch.zeros([signal.shape[0], len(reserve_list), signal.shape[2]])
    
    for idx, ch in enumerate(reserve_list):
        signal_filtered[:,idx,:] = signal[:,ch,:]
    
    return signal_filtered

def augment(signal, target, increase_size):
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
#     return (torch.cat((signal, tmp), 0)