import os
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

def plot_channel(tr_input, tr_target, ch = 2, num = 4):

    for i in range(num):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

        sample = tr_input[2*i,ch,:].numpy()  
    #     sample = tr_input[0,2*i,:].numpy()
        ax1.plot(sample)
        ax1.set_title(tr_target[2*i][0])

        sample = tr_input[2*i+1,ch,:].numpy()
    #     sample = tr_input[0,2*i+1,:].numpy()
        ax2.plot(sample)
        ax2.set_title(tr_target[2*i+1][0])
        plt.show()

def plot_fft(tr_input, tr_target, ch = 2, num = 4, target = 1):
    
    sample = tr_input[:num,ch,:].numpy()
    sample_target = tr_target[:num,0]
    
    for t, channel in zip(sample_target, sample):
        if int(t == target):
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
            sample_fft = np.fft.fft(channel)
            ax1.plot(channel)
            ax2.plot(abs(np.fft.fft(channel)[1:]))

            plt.show()

def plot_spectrogram(tr_input, tr_target, ch = 2, num = 4, target = 1, freq = 100, nseg = 10):
    
    sample = tr_input[:num,ch,:].numpy()
    sample_target = tr_target[:num,0]
    
    for t, channel in zip(sample_target, sample):
        if int(t == target):
            f, t, Sxx = signal.spectrogram(channel, freq, nperseg=nseg)
            plt.pcolormesh(t, f, Sxx)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

def plot_pair(before, after, ch = 2, num = 4):
    sample_before = before[:num,ch,:].numpy()
    sample_after = after[:num,ch,:].numpy()
    
    for channel1, channel2 in zip(sample_before, sample_after):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6), sharey=True)
        ax1.plot(channel1)
        ax2.plot(channel2)
        
        plt.show()

def plot_avg(signal, target, ch):
    sig1 = np.zeros(signal[0,0,:].shape)
    sig2 = np.zeros(signal[0,0,:].shape)
    num1, num2 = 0,0
    
    for t, data in zip(target[:,0], signal[:,ch,:]):
#         print(int(t==1))
        if int(t == 1):
            sig1 += data.numpy()
            num1+=1
        else:
            sig2 += data.numpy()
            num2+=1
    
    sig1/=num1
    sig2/=num2
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6), sharey=True)
    ax1.plot(sig1)
    ax1.set_title('channel' + str(ch))
    ax2.plot(sig2)
    ax2.set_title('channel' + str(ch))
    plt.show()