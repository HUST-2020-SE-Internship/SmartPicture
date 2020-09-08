import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import torchvision

# 所有label集合
labels_t = []
# 同一label的image名称集合
image_names = []
# read labels
with open('.\\data\\tiny-imagenet-200\\wnids.txt') as wnid:
    for line in wnid:
        labels_t.append(line.strip('\n'))

print("=== 读取训练集标签完毕")

# read image name by label
for label in labels_t:
    txt_path = '.\\data\\tiny-imagenet-200\\train\\' + label + '\\' + label + '_boxes.txt'
    image_name = []
    with open(txt_path) as txt:
        for line in txt:
            image_name.append(line.strip('\n').split('\t')[0])
    image_names.append(image_name)
labels = np.arange(200)

print("=== 读取训练集图像名称完毕")

# read test images labels
# 所有labels名称
val_labels_t = []
# 所有labels对应的序号
val_labels = []
# 所有image名称
val_names = []
with open('.\\data\\tiny-imagenet-200\\val\\val_annotations.txt') as txt:
    for line in txt:
        val_names.append(line.strip('\n').split('\t')[0])
        val_labels_t.append(line.strip('\n').split('\t')[1])
for i in range(len(val_labels_t)):
    for i_t in range(len(labels_t)):
        if val_labels_t[i] == labels_t[i_t]:
            val_labels.append(i_t)
val_labels = np.array(val_labels)

print("=== 读取测试集标签完毕")

COUNT = 10
DOWNLOAD = False
BATCH_SIZE = 1
LR = 0.00001
EPOCH = 40
USE_GPU = True
NET_PATH = "./net_tiny_image.pkl"


class TinyDataSet(Dataset):
    def __init__(self, dataType, transform):
        self.dataType = dataType
        if dataType == 'train':
            i = 0
            self.images = []
            print("=== 正在生成训练数据集...")
            for label in labels_t:
                print("=== 正在提取标签%s的数据" % label)
                image = []
                for image_name in image_names[i]:
                    image_path = os.path.join('.\\data\\tiny-imagenet-200\\train', label, 'images', image_name)
                    image.append(cv2.imread(image_path))
                self.images.append(image)
                i = i + 1
                if i >= COUNT:
                    break
            self.images = np.array(self.images)
            self.images = self.images.reshape(-1, 64, 64, 3)
        elif dataType == 'val':
            self.val_images = []
            print("=== 正在生成测试数据集...")
            for val_image in val_names:
                print("=== 正在获取测试集名称为%s的图片" % val_image)
                val_image_path = os.path.join('.\\data\\tiny-imagenet-200\\val\\images', val_image)
                self.val_images.append(cv2.imread(val_image_path))
            self.val_images = np.array(self.val_images)
        self.transform = transform

    def __getitem__(self, index):
        label = []
        image = []
        if self.dataType == 'train':
            label = index // 500
            image = self.images[index]
        if self.dataType == 'val':
            label = val_labels[index]
            image = self.val_images[index]
        return label, self.transform(image)

    def __len__(self):
        len = 0
        if self.dataType == 'train':
            len = self.images.shape[0]
        if self.dataType == 'val':
            len = self.val_images.shape[0]
        return len


train_dataset = TinyDataSet('train', transform=torchvision.transforms.ToTensor())
# val_dataset = TinyDataSet('val', transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class VGGNet(nn.Module):
    def __init__(self, num_classes, init_weights=True):
        super(VGGNet, self).__init__()
        # 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 16
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 8
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 4
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 2
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier2 = nn.Linear(1000, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# if __name__ == "__main__":
#     image_path = "D:/pythonfile/SmartPicture/data/tiny-imagenet-200/train/n02124075/images/n02124075_14.JPEG"
#     print(image_path)
#     image = cv2.imread(image_path)
#     # image = cv2.resize(image, (64, 64), cv2.INTER_AREA)
#     img = np.transpose(image, (2, 0, 1))
#     # print(type(img))
#     # print(img.shape)
#     img = torch.FloatTensor(img)
#     img = img.unsqueeze(0)
#     # print(type(img))
#     # print(img.shape)
#     device = torch.device("cuda:0")
#     print("Loading Net...")
#     net = torch.load("./net_tiny_image.pkl").to(device)
#     img = img.to(device)
#     print("Predicting...")
#     outputs = net(img)
#     _, predicted = torch.max(outputs.data, 1)
#     result = predicted[0].cpu().numpy()
#     print("result: ", result)

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

correct = 0
total = 0

for i, data in enumerate(train_dataloader):
    labels, inputs = data
    if USE_GPU:
        inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels.long()).sum().item()

    if i % 100 == 0:
        print('Accuracy of the network on the %d test images: %0.2f %%' % (
            i, 100 * correct / total))

    print(outputs.data)
    print("predicted: ", predicted, "labels: ", labels)
    img = inputs[0]
    img = img.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.show()
