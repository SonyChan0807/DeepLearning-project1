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
import torch.optim

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


def train_model(model, train_input, train_target, tr_target_onehot, train_mini_batch_size, test_input, test_target, te_target_onehot, test_mini_batch_size, epoch):
              
    """Function to train the model given the training and validation set. 
    Args:
        model                                     : class containing PyTorch model and the forward pass method (see nn_models.py)
        
        train_input           (torch.FloatTensor) : train samples with dimension (number of train samples x rows x columns)
        train_target          (torch.FloatTensor) : train categorical labels with dimension (number of test samples x 1)
        train_target_onehot   (torch.FloatTensor) : train onehot labels with dimension (number of test samples x no. of classes)
        train_mini_batch_size (int)               : size of training batch per iteration
        
        test_input            (torch.FloatTensor) : test samples with dimension (number of test samples x rows x columns)
        test_target           (torch.FloatTensor) : test categorical labels with dimension (number of test samples x 1)
        test_target_onehot    (torch.FloatTensor) : test one-hot labels with dimension (number of test samples x no. of classes)
        test_mini_batch_size  (int)               : size of test batch per iteration
        
    Returns:
        model                                     : class containing the trained PyTorch model
        tr_acc_all                                : training accuracies obtained at each epoch
        te_acc_all                                : test accuracies obtained at each epoch
        
    """  
                
    # initialize empty lists to collect
    # train and test accuracies and losses
    tr_loss_all = []
    te_loss_all = [] 
    tr_acc_all = []
    te_acc_all = []     
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Iterate for epochs
    for e in range(0, epoch):
        
        tr_acc  = 0
        te_acc  = 0        
        tr_loss  = 0
        te_loss  = 0

        # iterate through training set
        for b in range(0, np.shape(train_input)[0], train_mini_batch_size):
            if b + train_mini_batch_size > train_input.size(0):
                shift = train_input.size(0) - b
            else:
                shift = train_mini_batch_size

            # feedforward and compute loss
            output = model(train_input.narrow(0, b, shift), True)
            loss = criterion(output, train_target.narrow(0, b, shift))

            tr_loss = tr_loss + loss.data[0]
            
            # backpropagate
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        # iterate through test set
        for b in range(0, np.shape(test_input)[0], test_mini_batch_size):
            if b + test_mini_batch_size > test_input.size(0):
                shift = test_input.size(0) - b
            else:
                shift = test_mini_batch_size

            # feedforward and compute loss
            output = model(train_input.narrow(0, b, shift), True)
            loss = criterion(output, test_target.narrow(0, b, shift))

            te_loss = te_loss + loss.data[0]      
        # compute accuracy
        tr_acc  = compute_accuracy(model, train_input, tr_target_onehot,  train_mini_batch_size)
        te_acc  = compute_accuracy(model, test_input,  te_target_onehot,  test_mini_batch_size)
        print('epoch {:d} tr loss {:0.2f} val loss {:0.2f} tr acc {:f} val acc {:f}'.format(e, tr_loss, te_loss, tr_acc, te_acc))
               
        # accumulate train, val loss and accuracies   
        tr_acc_all.append(tr_acc)
        te_acc_all.append(te_acc)      
    
    return tr_acc_all, te_acc_all #, tr_loss_all, te_loss_all

# compute accuracy
def compute_accuracy(model, input, target_onehot, mini_batch_size, mode = False):

    """Function to compute accuracy
    Args:
        model                                     : Class containing PyTorch model and the forward pass method (see nn_models.py)        
        input                 (torch.FloatTensor) : test samples with dimension (number of train samples x rows x columns)
        target_onehot         (torch.FloatTensor) : test onehot labels with dimension (number of test samples x no. of classes)
        train_target_onehot   (torch.FloatTensor) : train onehot labels with dimension (number of test samples x no. of classes)
        mini_batch_size       (int)               : size of batch per iteration
        
    Returns:
        accuracy              (float)             : accuracy of the model in percentage points
    """ 
    
    accuracy = 0

    for b in range(0, input.size(0), mini_batch_size):
        if b + mini_batch_size > input.size(0):
            shift = input.size(0) - b
        else:
            shift = mini_batch_size

        output = model(input.narrow(0, b, shift), mode)

        _, predicted_classes = output.data.max(1)
        for k in range(0, shift):
            if(target_onehot.data[b + k, predicted_classes[k]] >= 0):
                accuracy = accuracy + 1

    return accuracy/input.size(0)
        
def compute_nb_errors(model, input, target, mini_batch_size):

    """Function to compute accuracy
    Args:
        model                                     : Class containing PyTorch model and the forward pass method (see nn_models.py)        
        input                 (torch.FloatTensor) : test samples with dimension (number of train samples x rows x columns)
        target_onehot         (torch.FloatTensor) : test onehot labels with dimension (number of test samples x no. of classes)
        train_target_onehot   (torch.FloatTensor) : train onehot labels with dimension (number of test samples x no. of classes)
        mini_batch_size       (int)               : size of batch per iteration
        
    Returns:
        error                 (int)               : number of misclassified samples
    """ 
    
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(0, mini_batch_size):
            if target.data[b + k, predicted_classes[k]] < 0:
                nb_errors = nb_errors + 1

    return nb_errors