import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import math

class GS2(nn.Module):
    def __init__(self, scale_factor, cr = 0.25):
        upsample_block_num = int(math.log(scale_factor, 2))
        self.cr = cr
        super(GS2, self).__init__()
        # self.block1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=9, padding=4),
        #     nn.PReLU()
        # )
        self.block1 = nn.Sequential(
            nn.Conv2d(3,int( (3 + 64) * self.cr ),kernel_size = 1,stride=1),
            nn.BatchNorm2d( int( (3 + 64) * self.cr ) ),
            nn.PReLU(),

            nn.Conv2d(int( (3 + 64) * self.cr ), int( (3 + 64) * self.cr ),kernel_size = (1,9),stride=1,padding=(0,4)),
            nn.BatchNorm2d( int( (3 + 64) * self.cr ) ),
            nn.PReLU(),

            nn.Conv2d( int( (3 + 64) * self.cr ) , int( (3 + 64) * self.cr ) ,kernel_size = (9,1),stride=1,padding=(4,0)),
            nn.BatchNorm2d(int( (3 + 64) * self.cr )),
            nn.PReLU(),

            nn.Conv2d( int( (3 + 64) * self.cr ), 64,kernel_size = 1,stride=1),
            # nn.BatchNorm2d(34),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]

        block8.append(nn.Conv2d(64, int((64+3) * cr), kernel_size=1, stride=1))
        block8.append(nn.BatchNorm2d(int((64+3) * cr)))
        block8.append(nn.PReLU())
        block8.append(nn.Conv2d(int((64+3) * cr), int((64+3) * cr), kernel_size=(1,9), padding=(0,4)))
        block8.append(nn.BatchNorm2d(int((64+3) * cr)))
        block8.append(nn.PReLU())
        block8.append(nn.Conv2d(int((64+3) * cr), int((64+3) * cr), kernel_size=(9,1), padding=(4,0)))
        block8.append(nn.BatchNorm2d(int((64+3) * cr)))
        block8.append(nn.PReLU())
        block8.append(nn.Conv2d(int((64+3) * cr), 3, kernel_size=1, stride=1))
        self.block8 = nn.Sequential(*block8)


        # block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class DS2(nn.Module):
    def __init__(self):
        super(DS2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),


            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.prelu = nn.PReLU()
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)

#     def forward(self, x):
#         residual = self.conv1(x)
#         residual = self.bn1(residual)
#         residual = self.prelu(residual)
#         residual = self.conv2(residual)
#         residual = self.bn2(residual)
#         return x + residual


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        hid = int(channels * 0.25)
        self.CP_block = nn.Sequential(
            nn.Conv2d(channels, hid, kernel_size=1),
            nn.BatchNorm2d(hid),
            nn.PReLU(),

            nn.Conv2d(hid, hid, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.BatchNorm2d(hid),
            nn.PReLU(),

            nn.Conv2d(hid, hid, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(hid),
            nn.PReLU(),

            nn.Conv2d(hid, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),


            nn.Conv2d(channels, hid, kernel_size=1),
            nn.BatchNorm2d(hid),
            nn.PReLU(),

            nn.Conv2d(hid, hid, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.BatchNorm2d(hid),
            nn.PReLU(),

            nn.Conv2d(hid, hid, kernel_size=(3,1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(hid),
            nn.PReLU(),

            nn.Conv2d(hid, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels)
        )
        

    def forward(self, x):
        residual = self.CP_block(x)
        return x + residual



class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale, cr = 0.25):
        super(UpsampleBLock, self).__init__()
        out_channels = in_channels * up_scale ** 2
        hid = int((in_channels + out_channels) * cr)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, hid, kernel_size=1),
        #     nn.BatchNorm2d(hid),
        #     nn.PReLU(),
        #     nn.Conv2d(hid, hid, kernel_size=(1,3), padding=(0,1)),
        #     nn.BatchNorm2d(hid),
        #     nn.PReLU(),
        #     nn.Conv2d(hid, hid, kernel_size=(3,1), padding=(1,0)),
        #     nn.BatchNorm2d(hid),
        #     nn.PReLU(),
        #     nn.Conv2d(hid, out_channels, kernel_size=1, stride=1)
        # )
        self.conv1 = nn.Conv2d(in_channels, hid, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hid)
        self.ac1 = nn.PReLU()
        self.conv2 = nn.Conv2d(hid, hid, kernel_size=(1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(hid)
        self.ac2 = nn.PReLU()
        self.conv3 = nn.Conv2d(hid, hid, kernel_size=(3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(hid)
        self.ac3 = nn.PReLU()
        self.conv4 = nn.Conv2d(hid, out_channels, kernel_size=1, stride=1)

        # self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ac1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.ac2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.ac3(x)
        x = self.conv4(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


# class UpsampleBLock(nn.Module):
#     def __init__(self, in_channels, up_scale):
#         super(UpsampleBLock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
#         self.pixel_shuffle = nn.PixelShuffle(up_scale)
#         self.prelu = nn.PReLU()

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.pixel_shuffle(x)
#         x = self.prelu(x)
#         return x



