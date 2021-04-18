import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(32, 3, padding=1)  # 32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 32
        s = s // 2  # 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 16
        s = s // 2  # 8
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 8
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        s = compute_conv_output_size(s, 3, padding=1)  # 8

        s = s // 2  # 4
        self.fc1 = nn.Linear(s * s * 128, 256)  # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)

        self.last = torch.nn.Linear(256, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x, get_inter=False, before_fc=False):
        act1 = self.relu(self.conv1(x))
        act2 = self.relu(self.conv2(act1))
        h = self.drop1(self.MaxPool(act2))
        act3 = self.relu(self.conv3(h))
        act4 = self.relu(self.conv4(act3))
        h = self.drop1(self.MaxPool(act4))
        act5 = self.relu(self.conv5(h))
        act6 = self.relu(self.conv6(act5))
        h = self.drop1(self.MaxPool(act6))
        h = h.view(x.shape[0], -1)
        act7 = self.relu(self.fc1(h))
        # h = self.drop2(act7)
        y=self.last(act7)

        if get_inter:
            if before_fc:
                return act6, y
            else:
                return act7, y
        else:
            return y


def compute_conv_output_size(l_in, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((l_in + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))
