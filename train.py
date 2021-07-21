from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
from model import lenet

def train(model, criterion, optimizer, train_loader, validation_loader, epochs):
    i=100
    stuff = {'training_loss': [], 'validation_loss': []}

    for epoch in range(epochs):
        for i, (x,y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            stuff['training_loss'].append(loss.data.item())
        
        correct = 0
        for (x,y) in validation_loader:
            z = model(x.view(-1, 28*28))
            _, label = torch.max(z)
            correct += (label == y).sum().item()
        
        accuracy = 100 * (correct / len(validation_dataset))
        stuff['validation_accuracy'].append(accuracy)
    return stuff


if __name__ == '__main__':
    model = lenet.LeNet()
    
    criterion = nn.CrossEntropyLoss()
    
    l_rate = 0.01
    optimizer = optim.Adam(model.parameters(), lr=l_rate)

    train_dataset = datasets.MNIST(
        root='datasets/mnist', train=True, download=False)
    validation_dataset = datasets.MNIST(
        root='datasets/mnist', train=False, download=False)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=64, shuffle=True)
    
    dataset_sizes = {
        'train':len(train_dataset), 'val':len(validation_dataset)
        }

    print(dataset_sizes)
    for batch_ndx, sample in enumerate(train_loader):
        print(sample.inp.is_pinned())
        print(sample.tgt.is_pinned())

    # train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     train_loader=train_loader,
    #     validation_loader=validation_loader,
    #     epochs=10
    # )
