import cv2
import imutils
from torch import nn
import torch
import numpy as np


# class VGGNet(nn.Module):
#     def __init__(self):
#         super(VGGNet, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=3,
#                 out_channels=16,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=32,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=48,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=48,
#                 out_channels=64,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=64,
#                 out_channels=96,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=96,
#                 out_channels=128,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             ),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.out = nn.Linear(128 * 4 * 4, 10)
#         self.out1 = nn.Linear(64 * 8 * 8, 100)
#         self.out2 = nn.Linear(100, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.view(x.size(0), -1)
#         # output = self.out1(x)
#         # output = self.out2(output)
#         output = self.out(x)
#         return output


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


def resize_crop_image(image, target_width, target_height):
    (h, w) = image.shape[:2]
    dH = 0
    dW = 0
    if w < h:
        image = imutils.resize(image, width=target_width,
                               inter=cv2.INTER_AREA)
        dH = int((image.shape[0] - target_height) / 2.0)
    else:
        image = imutils.resize(image, height=target_height,
                               inter=cv2.INTER_AREA)
        dW = int((image.shape[1] - target_width) / 2.0)
    (h, w) = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    image_path = "C:/Users/xhw/Pictures/TestData/cat3.jpg"
    image = cv2.imread(image_path)
    img = cv2.resize(image, (32, 32), cv2.INTER_AREA)
    # img = resize_crop_image(image, 32, 32)
    img = np.transpose(img, (2, 0, 1))
    # print(type(img))
    # print(img.shape)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)
    # print(type(img))
    # print(img.shape)
    device = torch.device("cuda:0")
    print("Loading Net...")
    net = torch.load("./net_vgg_train_3.pkl").to(device)
    img = img.to(device)
    print("Predicting...")
    outputs = net(img)
    _, predicted = torch.max(outputs.data, 1)
    result = predicted[0].cpu().numpy()
    print("The Image is => ", end="")
    if result == 0:
        print("airplane")
    elif result == 1:
        print("automobile")
    elif result == 2:
        print("bird")
    elif result == 3:
        print("cat")
    elif result == 4:
        print("deer")
    elif result == 5:
        print("dog")
    elif result == 6:
        print("frog")
    elif result == 7:
        print("horse")
    elif result == 8:
        print("ship")
    elif result == 9:
        print("thunk")
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # cv2.imshow("input", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
