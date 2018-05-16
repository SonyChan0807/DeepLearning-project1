import torch
from torch import nn
from torch.autograd import Variable # to initialize LSTM hidden states

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,13))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(28,7))
        self.linear = nn.Linear(96, 2)
        
    def forward(self, x, mode=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, x.size(1) * x.size(3))        
        x = self.linear(x)
        
        return x

class LSTM(nn.Module):
    def __init__(self, feature_dim, hidden_size, batch_size):
        super(LSTM, self).__init__()
                
        # single layer lstm
        self.lstm = nn.LSTM(feature_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.h0 = Variable(torch.zeros(1, batch_size, hidden_size)) 
        self.c0 = Variable(torch.zeros(1, batch_size, hidden_size))
        
        # fc layer
        self.fc1 = nn.Linear(hidden_size, 2)    
                
    def forward(self, x, mode=False):
                    
        output, hn = self.lstm(x, (self.h0,self.c0))  
        output = self.fc1(output[:,-1,:]) # output at last layer
        
        return output