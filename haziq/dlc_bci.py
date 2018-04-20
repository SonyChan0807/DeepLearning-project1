# This is distributed under BSD 3-Clause license

import torch
import numpy
import os
import errno

from six.moves import urllib

## remove

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

## 

def tensor_from_file(root, filename,
                     base_url = 'https://documents.epfl.ch/users/f/fl/fleuret/www/data/bci'):

    file_path = os.path.join(root, filename)

    if not os.path.exists(file_path):
        try:
            os.makedirs(root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = base_url + '/' + filename

        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            f.write(data.read())

    return torch.from_numpy(numpy.loadtxt(file_path))

def load(root, train = True, download = True, one_khz = False):
    """
    Args:

        root (string): Root directory of dataset.

        train (bool, optional): If True, creates dataset from training data.

        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

        one_khz (bool, optional): If True, creates dataset from the 1000Hz data instead
            of the default 100Hz.

    """

    nb_electrodes = 28

    if train:

        if one_khz:
            dataset = tensor_from_file(root, 'sp1s_aa_train_1000Hz.txt')
        else:
            dataset = tensor_from_file(root, 'sp1s_aa_train.txt')

        input = dataset.narrow(1, 1, dataset.size(1) - 1)
        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = dataset.narrow(1, 0, 1).clone().view(-1).long()

    else:

        if one_khz:
            input = tensor_from_file(root, 'sp1s_aa_test_1000Hz.txt')
        else:
            input = tensor_from_file(root, 'sp1s_aa_test.txt')
        target = tensor_from_file(root, 'labels_data_set_iv.txt')

        input = input.float().view(input.size(0), nb_electrodes, -1)
        target = target.view(-1).long()

    return input, target

def train_model(model, train_input, train_target, train_mini_batch_size, test_input, test_target, test_mini_batch_size, epoch):
        
    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    tr_loss_all = []
    te_loss_all = []
    eta = 1e-2

    for e in range(0, epoch):
        
        tr_loss = 0
        te_loss = 0
                
        # iterate through training set
        for b in range(0, train_input.size(0), train_mini_batch_size):
            
            # feedforward and compute loss
            output = model(train_input.narrow(0, b, train_mini_batch_size), True)
            loss = criterion(output, train_target.narrow(0, b, train_mini_batch_size))
            tr_loss = tr_loss + loss.data[0]
            
            # backpropagate to compute derivatives then
            # update the weights by subtracting the negative of the gradient
            model.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.data.sub_(eta * p.grad.data)
            
        # iterate through test set
        for b in range(0, test_input.size(0), test_mini_batch_size):
            
            # feedforward and compute loss
            output = model(test_input.narrow(0, b, test_mini_batch_size), False)
            loss = criterion(output, test_target.narrow(0, b, test_mini_batch_size))
            te_loss = te_loss + loss.data[0]        
            
        print('epoch {:d} tr loss {:0.2f} te loss {:0.2f}'.format(e, tr_loss, te_loss))
        tr_loss_all.append(tr_loss)
        te_loss_all.append(te_loss)
      
    return tr_loss_all, te_loss_all
        
def compute_nb_errors(model, input, target, mini_batch_size):

    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(0, mini_batch_size):
            if target.data[b + k, predicted_classes[k]] < 0:
                nb_errors = nb_errors + 1

    return nb_errors