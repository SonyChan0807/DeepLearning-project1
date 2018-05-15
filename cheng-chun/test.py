import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

# customized libraries
import dlc_bci as bci
import plot_lib as plib
import preprocess as prep
from nn_models import ConvNet2
from dlc_practical_prologue import *

if __name__ == "__main__":

	# load dataset
	tr_input, tr_target = bci.load("bci", train=True, one_khz=False)
	te_input, te_target = bci.load("bci", train=False, one_khz=False)

	# create outputs with one hot encoding
	tr_target_onehot = convert_to_one_hot_labels(tr_input, tr_target)
	te_target_onehot = convert_to_one_hot_labels(te_input, te_target)
	
	# convert output to variable
	tr_target_onehot = Variable(tr_target_onehot)
	te_target_onehot = Variable(te_target_onehot)
	tr_target_org = Variable(tr_target)
	te_target = Variable(te_target)

	# normalization
	tr_input = torch.nn.functional.normalize(tr_input, p=2, dim=0) 
	te_input = torch.nn.functional.normalize(te_input, p=2, dim=0) 

	# Convert to 4D tensor [dataset size, number of channels, rows, cols]
	tr_input = tr_input[:, np.newaxis, :, :]
	te_input = te_input[:, np.newaxis, :, :]

	# convert input to variable
	tr_input = Variable(tr_input)
	te_input = Variable(te_input)

	model = ConvNet2()
	# print(torch.load('models/best4.pth'))
	model.load_state_dict(torch.load('models/best_conv2.pth'))

	nb_errors = bci.compute_nb_errors(model, te_input, te_target_onehot, 10)
	print('accuracy = {:0.2f}'.format((te_input.shape[0]-nb_errors)/te_input.shape[0]))
