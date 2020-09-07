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
BATCH_SIZE = 8
LR = 0.00001
EPOCH = 5
USE_GPU = True
NET_PATH = "./net_vgg_train_3.pkl"


cifar_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=DOWNLOAD, transform=torchvision.transforms.ToTensor())
cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=DOWNLOAD, transform=torchvision.transforms.ToTensor())

trainloader = Data.DataLoader(cifar_train, batch_size=BATCH_SIZE, shuffle=True)
testloader = Data.DataLoader(cifar_test, batch_size=BATCH_SIZE, shuffle=True)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        # 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 2
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x


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
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
# optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)

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
