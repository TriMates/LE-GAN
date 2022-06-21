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


class GS1(nn.Module):
    def __init__(self, opt):
        super(GS1, self).__init__()
        nz = opt.nz
        ngf = opt.ngf
        self.ngpu = opt.ngpu
        cr = 0.5#compress rate
        self.main = nn.Sequential(
            nn.ReflectionPad2d((0,1,0,1)),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, int((ngf * 8 + nz) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ngf * 8 + nz) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 8 + nz) * cr), int((ngf * 8 + nz) * cr), kernel_size=(1, 4), stride=(1,1), bias=False),
            nn.BatchNorm2d(int((ngf * 8 + nz) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 8 + nz) * cr), int((ngf * 8 + nz) * cr), kernel_size=(4, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ngf * 8 + nz) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 8 + nz) * cr), ngf * 8, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, int((ngf * 8 + ngf * 4) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ngf * 8 + ngf * 4) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 8 + ngf * 4) * cr), int((ngf * 8 + ngf * 4) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(int((ngf * 8 + ngf * 4) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 8 + ngf * 4) * cr), int((ngf * 8 + ngf * 4) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(int((ngf * 8 + ngf * 4) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 8 + ngf * 4) * cr), ngf * 4, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, int((ngf * 4 + ngf * 2) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ngf * 4 + ngf * 2) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 4 + ngf * 2) * cr), int((ngf * 4 + ngf * 2) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(int((ngf * 4 + ngf * 2) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 4 + ngf * 2) * cr), int((ngf * 4 + ngf * 2) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(int((ngf * 4 + ngf * 2) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 4 + ngf * 2) * cr), ngf * 2, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),


            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, int((ngf * 2 + ngf) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ngf * 2 + ngf) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 2 + ngf) * cr), int((ngf * 2 + ngf) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(int((ngf * 2 + ngf) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 2 + ngf) * cr), int((ngf * 2 + ngf) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(int((ngf * 2 + ngf) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf * 2 + ngf) * cr),     ngf, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf, int((ngf + 3) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((ngf + 3) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf + 3) * cr), int((ngf + 3) * cr), kernel_size=(1, 4), stride=(1, 1), padding=(1, 2), bias=False),
            nn.BatchNorm2d(int((ngf + 3) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf + 3) * cr), int((ngf + 3) * cr), kernel_size=(4, 1), stride=(1, 1), padding=(2, 1), bias=False),
            nn.BatchNorm2d(int((ngf + 3) * cr)),
            nn.ReLU(True),
            nn.ConvTranspose2d(int((ngf + 3) * cr), 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        
        output = self.main(input)
        return output




class DS1(nn.Module):
    def __init__(self, opt):
        super(DS1, self).__init__()
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
            nn.Conv2d(ndf * 8, int((1 + ndf * 8) * cr), kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((1 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int((1 + ndf * 8) * cr), int((1 + ndf * 8) * cr), kernel_size=(1, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((1 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int((1 + ndf * 8) * cr), int((1 + ndf * 8) * cr), kernel_size=(4, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(int((1 + ndf * 8) * cr)),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(int((1 + ndf * 8) * cr), 1, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        # if input.is_cuda and self.ngpu > 1:
        #     output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        # else:
        #     output = self.main(input)
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)