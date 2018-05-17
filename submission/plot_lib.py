import os
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.signal import hilbert, chirp
from scipy import signal

# customized library
import dlc_bci as bci
from dlc_practical_prologue import *


def plot_channel(inputs, targets, ch = 2, num = 4):
    """Function to plot the time series given number of channels and number of data
    Args:
        inputs        (torch.FloatTensor): Matrix with size N x C x L
        targets       (torch.LongTensor) : Array with length N  
        ch           (int)              : number of channels to print
        num          (int)              : number of data to print
    """
    for i in range(num):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))

        sample = inputs[2*i,ch,:].numpy()  
        ax1.plot(sample)
        ax1.set_title(targets[2*i][0])

        sample = inputs[2*i+1,ch,:].numpy()
        ax2.plot(sample)
        ax2.set_title(targets[2*i+1][0])
        plt.show()

def plot_fft(inputs, targets, ch = 2, num = 4, target = 1):
    """Function to plot the spectrum of time series
    Args:
        inputs       (torch.FloatTensor): Matrix with size N x C x L
        targets      (torch.LongTensor) : Array with length N  
        ch           (int)              : number of channels to print
        num          (int)              : number of data to print
        target      (int)               : the target to be printed -> 0 or 1
    """    
    sample = inputs[:num,ch,:].numpy()
    sample_targets = targets[:num,0]
    
    for t, channel in zip(sample_targets, sample):
        if int(t == target):
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6))
            sample_fft = np.fft.fft(channel)
            ax1.plot(channel)
            ax2.plot(abs(np.fft.fft(channel)[1:]))

            plt.show()

def plot_spectrogram(inputs, targets, ch = 2, num = 4, target = 1, freq = 100, nseg = 10):
    """Function to plot the spectrogram of the time series
    Args:
        inputs        (torch.FloatTensor): Matrix with size N x C x L
        targets       (torch.LongTensor) : Array with length N  
        ch            (int)              : number of channels to print
        num           (int)              : number of data to print
        target        (int)              : the target to be printed -> 0 or 1
        freq          (int)              : frequency 
        nseg          (int)              : number of segments
    """

    sample = inputs[:num,ch,:].numpy()
    sample_targets = targets[:num,0]
    
    for t, channel in zip(sample_targets, sample):
        if int(t == target):
            f, t, Sxx = signal.spectrogram(channel, freq, nperseg=nseg)
            plt.pcolormesh(t, f, Sxx)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

def plot_pair(before, after, ch = 2, num = 4):
    """Function used to plot to compare the effects of preprocessing
    Args:
        before       (torch.FloatTensor): Matrix with size N x C x L1
        after        (torch.FloatTensor) : Array with length N x C x L2 
        ch           (int)              : number of channels to print
        num          (int)              : number of data to print
    """    
    sample_before = before[:num,ch,:].numpy()
    sample_after = after[:num,ch,:].numpy()
    
    for channel1, channel2 in zip(sample_before, sample_after):
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6), sharey=True)
        ax1.plot(channel1)
        ax2.plot(channel2)
        
        plt.show()

def plot_avg(signal, targets, ch):
    """Function used to plot the average of signals with targets 1 and 0
    Args:
        signal       (torch.FloatTensor): Matrix with size N x C x L
        targets       (torch.LongTensor) : Array with length N  
        ch           (int)              : number of channels to print
    """    

    sig1 = np.zeros(signal[0,0,:].shape)
    sig2 = np.zeros(signal[0,0,:].shape)
    num1, num2 = 0,0
    
    for t, data in zip(targets[:,0], signal[:,ch,:]):
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