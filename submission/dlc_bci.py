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

def train_model(model_name, model, train_input, train_target, train_mini_batch_size, val_input, val_target, test_input, test_target, epoch):
        
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    tr_loss_all = []
    te_loss_all = []
    va_loss_all = []

    eta = 1e-2
    # eta = 1e-3
    # optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adadelta(model.parameters(), lr = eta)

    path = os.getcwd() +'/models/' + model_name
    # path = '/home/cheng-chun-epfl/Dropbox/EPFL/course/MA2/deep learning/min-project/cheng-chun/models/' + model_name
    te_acc_max = 0
    va_acc_best = 0
    va_loss_best = 10
    tr_loss_best = 10
    te_acc_best = 0


    for e in range(0, epoch):
        
        # tr_loss = 0
        # te_loss = 0
        # val_loss = 0
        # iterate through training set
        for b in range(0, train_input.size(0), train_mini_batch_size):
            
            # feedforward and compute loss

            if b + train_mini_batch_size > train_input.size(0):
                shift = train_input.size(0) - b
            else:
                shift = train_mini_batch_size
            output = model(train_input.narrow(0, b, shift), True)
            loss = criterion(output, train_target.narrow(0, b, shift))
            # tr_loss = tr_loss + loss.data[0]
            
            # backpropagate to compute derivatives then
            # update the weights by subtracting the negative of the gradient
            model.zero_grad()
            loss.backward()
            # for p in model.parameters():
            #     p.data.sub_(eta * p.grad.data)
            optimizer.step()
            
        # iterate through test set
        # for b in range(0, test_input.size(0), test_mini_batch_size):
            
            # feedforward and compute loss
            # output = model(test_input.narrow(0, b, test_mini_batch_size), False)
            # loss = criterion(output, test_target.narrow(0, b, test_mini_batch_size))
            # te_loss = te_loss + loss.data[0]        

        # iterate through val set
        # for b in range(0, val_input.size(0), val_mini_batch_size):
            
            # feedforward and compute loss
            # output = model(val_input.narrow(0, b, val_mini_batch_size), False)
            # loss = criterion(output, val_target.narrow(0, b, val_mini_batch_size))
            # val_loss = val_loss + loss.data[0]  


        # print("epoch ", e)
        # # print('epoch {:d} tr loss {:0.2f} te loss {:0.2f}'.format(e, tr_loss, te_loss*3.16))
        # num_correct = np.sum((torch.max(F.softmax(model(train_input), 1), 1)[1] == train_target).data.numpy())
        # print('tr acc = {:0.2f}'.format(num_correct/train_target.shape[0]))
        
        # # print(num_correct/train_target.shape[0])
        # # if num_correct/train_target.shape[0] >= 0.85:
        # #     num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
        # #     te_acc = num_correct/test_target.shape[0]
        # #     print('epoch: ', e, ', te acc = {:0.2f}'.format(te_acc))
        # #     break
        
        # num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
        # te_acc = num_correct/test_target.shape[0]
        # print('te acc = {:0.2f}'.format(num_correct/test_target.shape[0]))

        # num_correct = np.sum((torch.max(F.softmax(model(val_input), 1), 1)[1] == val_target).data.numpy())
        # val_acc = num_correct/val_target.shape[0]
        # print('val acc = {:0.2f}'.format(val_acc))

        # if val_acc_max < val_acc:
        #     val_acc_max = val_acc


        output = model(test_input, False)
        loss = criterion(output, test_target)
        te_loss = loss.data[0]      

        output = model(val_input, False)
        loss = criterion(output, val_target)
        va_loss = loss.data[0]      
            
        output = model(train_input, False)
        loss = criterion(output, train_target)
        tr_loss = loss.data[0] 
        
        num_correct = np.sum((torch.max(F.softmax(model(val_input), 1), 1)[1] == val_target).data.numpy())
        va_acc = num_correct/val_target.shape[0]

        # if va_acc_best < va_acc:
        if va_loss_best > va_loss:

            va_loss_best = va_loss
            tr_loss_best = tr_loss
            num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
            te_acc_best = num_correct/test_target.shape[0]

            num_correct = np.sum((torch.max(F.softmax(model(train_input), 1), 1)[1] == train_target).data.numpy())
            tr_acc_best = num_correct/train_target.shape[0]
                    
            num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
            te_acc_best = num_correct/test_target.shape[0]

            va_acc_best = va_acc
            # num_correct = np.sum((torch.max(F.softmax(model(val_input), 1), 1)[1] == val_target).data.numpy())
            # val_acc_best = num_correct/val_target.shape[0]
        # if e == epoch - 1:
        if (e+1)%20 == 0:
            print('epoch {:d} tr loss {:0.2f}  val loss {:0.2f} te loss {:0.2f}'.format(e+1, tr_loss, va_loss, te_loss))

            num_correct = np.sum((torch.max(F.softmax(model(train_input), 1), 1)[1] == train_target).data.numpy())
            print('tr acc = {:0.2f}'.format(num_correct/train_target.shape[0]))
            
            # print(num_correct/train_target.shape[0])
            # if num_correct/train_target.shape[0] >= 0.85:
            #     num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
            #     te_acc = num_correct/test_target.shape[0]
            #     print('epoch: ', e, ', te acc = {:0.2f}'.format(te_acc))
            #     break
            
            # num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
            # te_acc = num_correct/test_target.shape[0]
            # print('te acc = {:0.2f}'.format(num_correct/test_target.shape[0]))

            num_correct = np.sum((torch.max(F.softmax(model(val_input), 1), 1)[1] == val_target).data.numpy())
            val_acc = num_correct/val_target.shape[0]
            print('val acc = {:0.2f}'.format(val_acc))
        
        # num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
        # te_acc = num_correct/test_target.shape[0]
        # print('te acc = {:0.2f}'.format(num_correct/test_target.shape[0]))


            # print('current best val acc: {:0.2f}'.format(val_acc_max))
        # if te_acc_max < te_acc:
            # torch.save(model.state_dict(), path)
            # print('save the current best: {:0.2f}'.format(te_acc))
            # te_acc_max = te_acc

        tr_loss_all.append(tr_loss)
        # te_loss_all.append(te_loss)
        va_loss_all.append(va_loss)

        # if va_loss <= 0.4:
        #     break

    # print('best model in this model: {:0.2f}'.format(te_acc_max))
    print('best')
    print('epoch {:d} tr loss {:0.2f}  val loss {:0.2f}'.format(e, tr_loss_best, va_loss_best))

    # num_correct = np.sum((torch.max(F.softmax(model(train_input), 1), 1)[1] == train_target).data.numpy())
    print('tr acc = {:0.2f}'.format(tr_acc_best))
            
    # num_correct = np.sum((torch.max(F.softmax(model(test_input), 1), 1)[1] == test_target).data.numpy())
    # te_acc = num_correct/test_target.shape[0]
    print('te acc = {:0.2f}'.format(te_acc_best))

    # num_correct = np.sum((torch.max(F.softmax(model(val_input), 1), 1)[1] == val_target).data.numpy())
    # val_acc = num_correct/val_target.shape[0]
    print('val acc = {:0.2f}'.format(va_acc_best))

    return tr_loss_all, va_loss_all, te_acc_best, va_acc_best, tr_acc_best

def train_model_haziq(model, train_input, train_target, tr_target_onehot, train_mini_batch_size, test_input, test_target, te_target_onehot, test_mini_batch_size, epoch):
              
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
            # print(shift)
            # feedforward and compute loss
            # output = model(train_input.narrow(0, b, train_mini_batch_size), True)
            # loss = criterion(output, train_target.narrow(0, b, train_mini_batch_size))

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
            # print(shift)
            # feedforward and compute loss
            # output = model(test_input.narrow(0, b, test_mini_batch_size), False)
            # loss = criterion(output, train_target.narrow(0, b, train_mini_batch_size))
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
def compute_accuracy(model, input, target, mini_batch_size, mode = False):

    accuracy = 0

    for b in range(0, input.size(0), mini_batch_size):
        if b + mini_batch_size > input.size(0):
            shift = input.size(0) - b
        else:
            shift = mini_batch_size

        # output = model(input.narrow(0, b, mini_batch_size), mode)
        output = model(input.narrow(0, b, shift), mode)

        _, predicted_classes = output.data.max(1)
        # for k in range(0, mini_batch_size):
        for k in range(0, shift):
            if(target.data[b + k, predicted_classes[k]] >= 0):
                accuracy = accuracy + 1

    return accuracy/input.size(0)
        
def compute_nb_errors(model, input, target, mini_batch_size):

    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = output.data.max(1)
        for k in range(0, mini_batch_size):
            if target.data[b + k, predicted_classes[k]] < 0:
                nb_errors = nb_errors + 1

    return nb_errors