import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

# customized libraries
import dlc_bci as bci
import plot_lib as plib
import preprocess as prep
from nn_models import ConvNet3, LSTM

if __name__ == "__main__":

    

    print("Loading the dataset......")
    # load dataset
    tr_input_org, tr_target_org = bci.load("bci", train=True, one_khz=False)
    te_input_org, te_target_org = bci.load("bci", train=False, one_khz=False)

    # normalization
    tr_input_org = torch.nn.functional.normalize(tr_input_org, p=2, dim=0) 
    te_input_org = torch.nn.functional.normalize(te_input_org, p=2, dim=0) 

    # create outputs with one hot encoding
    tr_target_onehot = prep.convert_to_one_hot_labels(tr_input_org, tr_target_org)
    te_target_onehot = prep.convert_to_one_hot_labels(te_input_org, te_target_org)

    # convert output to variable
    tr_target_onehot = Variable(tr_target_onehot)
    te_target_onehot = Variable(te_target_onehot)
    tr_target = Variable(tr_target_org)
    te_target = Variable(te_target_org)

    # Convert to 4D tensor [dataset size, number of channels, rows, cols]
    tr_input = tr_input_org[:, np.newaxis, :, :]
    te_input = te_input_org[:, np.newaxis, :, :]

    # convert input to variable
    tr_input = Variable(tr_input)
    te_input = Variable(te_input)

    print("Our best model: CNN......")
    # CNN Model
    # ----------
    model = ConvNet3()
    bci.train_model(model, tr_input, tr_target, tr_target_onehot, 10, te_input, te_target, te_target_onehot, 10, 200)

    nb_errors = bci.compute_nb_errors(model, te_input, te_target_onehot, 10)
    print('CNN accuracy = {:0.2f}'.format((te_input.shape[0]-nb_errors)/te_input.shape[0]))
    
    
    
    print("Other model: LSTM......")
    # LSTM Model
    # ----------
    model = LSTM(feature_dim = 28, hidden_size=25, batch_size=10)
    
    # Rearrange data
    tr_input = np.transpose(tr_input_org, (0,2,1))
    te_input = np.transpose(te_input_org, (0,2,1))
    
    # Append training samples since PyTorch LSTM requires a fixed batch size
    tr_input, tr_target = prep.replicate_samples(tr_input, tr_target_org, 4)
    
    # Prepare one-hot encoding for PyTorch computation of loss
    tr_target_onehot = prep.convert_to_one_hot_labels(tr_input, tr_target)
    
    # convert input to variable
    tr_input  = Variable(tr_input)
    tr_target = Variable(tr_target)
    tr_target_onehot = Variable(tr_target_onehot)
    te_input = Variable(te_input)
    
    # train LSTM
    tr_acc, te_acc = bci.train_model(model, tr_input, tr_target, tr_target_onehot, 10, te_input, te_target, te_target_onehot, 10, 200)
    
    nb_errors = bci.compute_nb_errors(model, te_input, te_target_onehot, 10)
    print('LSTM accuracy = {:0.2f}'.format((te_input.shape[0]-nb_errors)/te_input.shape[0]))
