import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.l5 = nn.Linear(in_features=400, out_features=120)
        self.l6 = nn.Linear(in_features=120, out_features=84)
        self.l7 = nn.Linear(in_features=84, out_features=10)

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.c1(x)
        x = torch.sigmoid(x)
        x = self.s2(x)

        x = self.c3(x)
        x = torch.sigmoid(x)
        x = self.s4(x)

        x = self.flatten(x)
        x = self.l5(x)
        x = torch.sigmoid(x)

        x = self.l6(x)
        x = torch.sigmoid(x)
        
        x = self.l7(x)
        x = torch.sigmoid(x)
        
        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    net = LeNet()
    input = torch.randn(1, 1, 28, 28)
    out = net.forward(input)
    print(out.shape)
