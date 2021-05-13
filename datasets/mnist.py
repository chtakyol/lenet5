import torchvision.datasets as datasets

# Downlad
# train_dataset = datasets.MNIST(root='mnist/', train=True, download=True)
# test_dataset = datasets.MNIST(root='mnist/', train=False, download=True)

# Load
train_dataset = datasets.MNIST(root='mnist/', train=True, download=False)
test_dataset = datasets.MNIST(root='mnist/', train=False, download=False)

dataset_sizes = {'train':len(train_dataset), 'test':len(test_dataset)}



print(dataset_sizes)