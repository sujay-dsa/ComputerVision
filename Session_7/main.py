import torch
from torch import optim
from torchsummary import summary


from sdmodel import MNISTNet
from sdmodel import MNISTNetGBN
from sdtransforms import SDLoader
from sdtransforms import MnistTransforms
from test import test_mnist
from train import train_mnist
from torchvision import datasets

sdtransform = MnistTransforms(1)
train = datasets.MNIST('./data', train=True, download=True, transform=sdtransform.mnist_train_transforms())
test = datasets.MNIST('./data', train=False, download=True, transform=sdtransform.mnist_test_transforms())

train_loader, test_loader = SDLoader().mnist_data_loaders(train, test)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

model = MNISTNet(0.9).to(device)
summary(model, input_size=(1, 28, 28))

model2 = MNISTNetGBN(0.9).to(device)
summary(model2, input_size=(1, 28, 28))



#regularizers = ['l1', 'l2', 'l1l2', 'gbn', 'gbnl1l2']
regularizers =['l1']

losses = {}
accuracies = {}

test_losses = {}
test_accuracies = {}

EPOCHS = 25
for regularizer in regularizers:
    model = None
    optimizer = None
    scheduler = None

    if regularizer == 'l1':
        model = MNISTNet(0.9).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.2)

    if regularizer == 'l2':
        model = MNISTNet(0.9).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.2)

    if regularizer == 'l1l2':
        model = MNISTNet(0.9).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.2)

    if regularizer == 'gbn':
        model = MNISTNetGBN(0.9).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.2)

    if regularizer == 'gbnl1l2':
        model = MNISTNetGBN(0.9).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=7, gamma=0.2)

    losses[regularizer] = []
    test_losses[regularizer] = []
    accuracies[regularizer] = []
    test_accuracies[regularizer] = []
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_mnist(model, device, train_loader, optimizer, epoch, regularizer, losses[regularizer], accuracies[regularizer])
        test_mnist(model, device, test_loader, test_losses[regularizer], test_accuracies[regularizer], regularizer)
        scheduler.step()
