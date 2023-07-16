import torch
import torch.nn as nn
import numpy as np

class ResBlock(nn.Module):
    def __init__(self,in_dim,out_dim) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim,out_dim,kernel_size=(3,3),stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_dim,out_dim,kernel_size=(3,3),stride=1,padding=1)
        self.relu = nn.ReLU()

    def forward(self,in_x):
        x = self.relu(self.conv1(in_x))
        x = self.conv2(x) + in_x

        return x

class NoiseNet(nn.Module):
    def __init__(self, is_train=False):
        super(NoiseNet, self).__init__()
        self.is_train = is_train

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv_seq = nn.Sequential(
            ResBlock(64,64),
            nn.MaxPool2d(2,2),
            ResBlock(64,64),
            nn.MaxPool2d(2,2),

            ResBlock(64,64),
            nn.MaxPool2d(2,2),
            ResBlock(64,64),
            nn.MaxPool2d(2,2),
        )
        
        self.fc_seq = nn.Sequential(
            nn.Linear(64 * 4 ** 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.crop_size = 128

    def forward(self, in_x):
        if self.is_train == False:
            h, w = in_x.shape[2:]
            x, y = np.random.randint(
                0, w - self.crop_size), np.random.randint(0, h - self.crop_size)
            in_x = in_x[:, :, y:y + self.crop_size, x:x + self.crop_size]

        x = self.conv1(in_x)
        x = self.conv_seq(x)

        x = nn.Flatten()(x)
        x = self.fc_seq(x)

        return x
