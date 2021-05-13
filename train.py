from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import lenet

def train():
    train_dataset = datasets.MNIST(
        root='datasets/mnist', train=True, download=False)
    test_dataset = datasets.MNIST(
        root='datasets/mnist', train=False, download=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
    dataset_sizes = {
        'train':len(train_dataset), 'test':len(test_dataset)
        }
    print(dataset_sizes)


if __name__ == '__main__':
    train()
    l_rate = 0.01
    model = lenet.LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=l_rate)
