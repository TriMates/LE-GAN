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
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(GS2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,34,kernel_size = 1,stride=1),
            nn.BatchNorm2d(34),
            nn.PReLU(),
            nn.Conv2d(34,34,kernel_size = (1,9),stride=1,padding=(0,4)),
            nn.BatchNorm2d(34),
            nn.PReLU(),
            nn.Conv2d(34,34,kernel_size = (9,1),stride=1,padding=(4,0)),
            nn.BatchNorm2d(34),
            nn.PReLU(),
            nn.Conv2d(34,64,kernel_size = 1,stride=1),
            # nn.BatchNorm2d(34),
            nn.PReLU()

            # nn.Conv2d(3, 64, kernel_size=9, padding=4),
            # nn.PReLU()
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
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
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

class DS2_c1(nn.Module):
    def __init__(self, opt):
        super(DS2_c1, self).__init__()
        self.ngpu = opt.ngpu
        ndf = opt.ndf
        cr = 0.5
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, int((ndf+3) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf+3) * cr), int((ndf+3) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(int((ndf+3) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf+3) * cr), int((ndf+3) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(int((ndf+3) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf+3) * cr), ndf, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, int((ndf + ndf*2) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ndf + ndf*2) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf + ndf*2) * cr), int((ndf + ndf*2) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(int((ndf + ndf*2) * cr)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int((ndf + ndf*2) * cr), int((ndf + ndf*2) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(int((ndf + ndf*2) * cr)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int((ndf + ndf*2) * cr), ndf * 2, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, int((ndf * 2 + ndf * 4) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ndf * 2 + ndf * 4) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf * 2 + ndf * 4) * cr), int((ndf * 2 + ndf * 4) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(int((ndf * 2 + ndf * 4) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf * 2 + ndf * 4) * cr), int((ndf * 2 + ndf * 4) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(int((ndf * 2 + ndf * 4) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf * 2 + ndf * 4) * cr), ndf * 4, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, int((ndf * 4 + ndf * 8) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ndf * 4 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int((ndf * 4 + ndf * 8) * cr), int((ndf * 4 + ndf * 8) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(int((ndf * 4 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf * 4 + ndf * 8) * cr), int((ndf * 4 + ndf * 8) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(int((ndf * 4 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((ndf * 4 + ndf * 8) * cr), ndf * 8, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, int((1 + ndf * 8) * cr), kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(int((1 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((1 + ndf * 8) * cr), int((1 + ndf * 8) * cr), kernel_size=(1, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((1 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int((1 + ndf * 8) * cr), int((1 + ndf * 8) * cr), kernel_size=(4, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((1 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int((1 + ndf * 8) * cr), 1, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     output = self.main(input)
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)



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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        hid = int(channels/2)
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
            nn.BatchNorm2d(channels),
            nn.PReLU(),
        )
        


    def forward(self, x):
        residual = self.CP_block(x)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x



