import torch
from torchvision import datasets, transforms
from config import BATCH_SIZE

transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST('../MNIST/MNIST_data/train/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = datasets.MNIST('../MNIST/MNIST_data/test/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)