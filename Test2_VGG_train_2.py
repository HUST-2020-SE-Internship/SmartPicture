import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import os

DOWNLOAD = False
BATCH_SIZE = 16
LR = 0.00001
EPOCH = 3
USE_GPU = True
NET_PATH = "./net_vgg_train_2.pkl"


cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=DOWNLOAD, transform=torchvision.transforms.ToTensor())
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=DOWNLOAD, transform=torchvision.transforms.ToTensor())

trainloader = Data.DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = Data.DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=True)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=48,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(128 * 4 * 4, 10)
        self.out1 = nn.Linear(64 * 8 * 8, 100)
        self.out2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        # output = self.out1(x)
        # output = self.out2(output)
        output = self.out(x)
        return output


net = None
if os.path.exists(NET_PATH):
    net = torch.load(NET_PATH)
    print("Load %s" % NET_PATH)
else:
    net = VGGNet()
    print("New Net")

device = None
if USE_GPU & torch.cuda.is_available():
    device = torch.device("cuda:0")
    net = net.to(device)
    print("Run With GPU")
else:
    USE_GPU = False
    print("Run With CPU")

loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=LR)
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)

correct = 0
total = 0

print("Start Training...")
for epoch in range(EPOCH):
    loss100 = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        if USE_GPU:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss100 += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss100 / 100))
            loss100 = 0.0
            print('Accuracy of the network: %0.2f %%' % (
                    100 * correct / total))
            correct = 0
            total = 0
            print("\n")


print("Done Training!")

torch.save(net, NET_PATH)
