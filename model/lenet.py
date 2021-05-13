import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(
            in_channels=1, out_channels=6, 
            kernel_size=5, stride=1, padding=2
            )
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(
            in_channels=6, out_channels=16,
            kernel_size=5
        )
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(
            in_channels=16, out_channels=120,
            kernel_size=5
        )
        self.l6 = nn.Linear(in_features=120, out_features=84)
        self.l7 = nn.Linear(in_features=84, out_features=10)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.c1(x)
        x = self.tanh(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.tanh(x)
        x = self.s4(x)
        x = self.c5(x)
        x = self.tanh(x)
        x = torch.flatten(x)
        x = self.l6(x)
        x = self.tanh(x)
        x = self.l7(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(0)
    net = LeNet()
    input = torch.randn(1, 1, 28, 28)
    out = net.forward(input)
    print(out)
