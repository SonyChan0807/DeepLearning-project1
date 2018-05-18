from torch import nn


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=28, out_channels=28, kernel_size=3)
        self.linear1 = nn.Linear(336, 336)
        self.output = nn.Linear(336, 2)        
        
    def forward(self, x, mode=False):
        
        # convolve
        x = self.conv1(x)
        x = F.max_pool1d(F.relu(x),4)

        x = x.view(-1, x.size(1) * x.size(2))
        x = self.linear1(x)
        x = F.dropout(x, p=0.6, training=mode)
        x = F.relu(x)        

        x = self.output(x)
        
        return x

class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=28, out_channels=28, kernel_size=7)
        self.linear1 = nn.Linear(224, 224)
        self.output = nn.Linear(224, 2)        
        
    def forward(self, x, mode=False):
        
        # convolve
        x = self.conv1(x)
        x = F.max_pool1d(F.relu(x),4)
        x = x.view(-1, x.size(1) * x.size(2))

        x = self.linear1(x)
        x = F.dropout(x, p=0.6, training=mode)
        x = F.relu(x)        

        x = self.output(x)
        x = F.dropout(x, p=0.6, training=mode)
        
        return x

class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,13))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(28,7))
        self.linear = nn.Linear(96, 2)
        
    def forward(self, x, mode=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, x.size(1) * x.size(3))        
        x = self.linear(x)
        
        return x

class ConvNet4(nn.Module):
    def __init__(self):
        super(ConvNet4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,11))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(28,5))
        self.linear = nn.Linear(96, 2)
        
    def forward(self, x, mode=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, x.size(1) * x.size(3))        
        x = self.linear(x)
        
        return x