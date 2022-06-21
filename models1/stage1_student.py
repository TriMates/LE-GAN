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

class GS1_c16(nn.Module):
    def __init__(self, cr = 0.25):
        super(GT1_c16_DE, self).__init__()

        self.init_size = 64 // 4
        self.l1 = nn.Sequential(nn.Linear(100 + 256, 128 * self.init_size ** 2))
        self.cr = cr

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            #nn.Conv2d(128, 128, 3, stride=1, padding=1),

            nn.Conv2d(128, int((128 + 128) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d( int((128 + 128) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( int((128 + 128) * cr), int((128 + 128) * cr), kernel_size=(1, 3), stride=(1,1), padding=(0,1), bias=False),
            nn.BatchNorm2d( int((128 + 128) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( int((128 + 128) * cr), int((128 + 128) * cr), kernel_size=(3, 1), stride=(1, 1), padding=(1,0), bias=False),
            nn.BatchNorm2d( int((128 + 128) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( int((128 + 128) * cr), 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            #nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.Conv2d(128, int((128 + 64) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d( int((128 + 64) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( int((128 + 64) * cr), int((128 + 64) * cr), kernel_size=(1, 3), stride=(1,1), padding=(0,1), bias=False),
            nn.BatchNorm2d( int((128 + 64) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( int((128 + 64) * cr), int((128 + 64) * cr), kernel_size=(3, 1), stride=(1, 1), padding=(1,0), bias=False),
            nn.BatchNorm2d( int((128 + 64) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d( int((128 + 64) * cr), 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),


            #nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Conv2d(64, int((64 + 3) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d( int((64 + 3) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( int((64 + 3) * cr), int((64 + 3) * cr), kernel_size=(1, 3), stride=(1,1), padding=(0,1), bias=False),
            nn.BatchNorm2d( int((64 + 3) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d( int((64 + 3) * cr), int((64 + 3) * cr), kernel_size=(3, 1), stride=(1, 1), padding=(1,0), bias=False),
            nn.BatchNorm2d( int((64 + 3) * cr) ),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d( int((64 + 3) * cr), 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img



class DS1(nn.Module):
    def __init__(self, cr = 0.25):
        super(DS1, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                #nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                nn.Conv2d(in_filters, int((in_filters + out_filters) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(int((in_filters + out_filters) * cr), int((in_filters + out_filters) * cr), kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
                nn.BatchNorm2d(int((in_filters + out_filters) * cr)),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(int((in_filters + out_filters) * cr), int((in_filters + out_filters) * cr), kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
                nn.BatchNorm2d(int((in_filters + out_filters) * cr)),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(int((in_filters + out_filters) * cr), out_filters, kernel_size=(1, 1), stride=(2, 2), bias=False),

                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out).view(-1)

        return validity