from torch import nn

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
