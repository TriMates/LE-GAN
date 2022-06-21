from __future__ import generators
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


import utils.img2dataset as i2d

from loss import GeneratorLoss

#教师网络GT1 GT2
from mdl4.stage1_teacher import GT1
from mdl4.model import Generator as GT2

#学生网络GS1 GS2 DS1 DS2
from mdl4.stage1_student import GS1
from mdl4.stage1_student import DS1
from mdl4.stage2_student import GS2,DS2

from lossv6 import GeneratorLoss
torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./result/pre/teachernetv7.3_1209/output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', default=42, type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--ckpt', default='./result/pre/teachernetv7.3_1209/ckpt',help='ckpt file path')
parser.add_argument('--card', type=str, default='0',help='select GPU:0 or 1')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.card
try:
    os.makedirs(opt.outf)
    os.makedirs(opt.ckpt)
except OSError:
    pass

try:
    os.makedirs(opt.outf+"/1/")
    os.makedirs(opt.outf+"/2/")
except OSError:
    pass


random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

import dataload
dataset = dataload.DataSet()

dl = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=2)



gT1 = GT1()
gT2 = GT2(1)
# 学生网络
gS1 = GS1()
gS2 = GS2(1)

dS1 = DS1()
dS2 = DS2()

if(opt.cuda):
    #教师网络
    gT1 = nn.DataParallel(gT1).cuda()
    gT2 = nn.DataParallel(gT2).cuda()
    #学生网络
    gS1 = nn.DataParallel(gS1).cuda()
    gS2 = nn.DataParallel(gS2).cuda()
    dS1 = nn.DataParallel(dS1).cuda()
    dS2 = nn.DataParallel(dS2).cuda()

gT1.load_state_dict(torch.load("./ckpt_T/netG1_Final.pth"))
gT2.load_state_dict(torch.load("./ckpt_T/netG2_Final.pth"))
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)



criterion = nn.MSELoss()
MSEcriterion = nn.MSELoss()
generator_criterion = GeneratorLoss(100, 1, 5).cuda()
SMTL1criterion = nn.SmoothL1Loss()

# fixed_noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()
real_label = 1.
fake_label = 0.


noise = None

Tensor = torch.cuda.FloatTensor





criterion = nn.MSELoss()
MSEcriterion = nn.MSELoss()
generator_criterion = GeneratorLoss().cuda()

real_label = 1.
fake_label = 0.

optimizerGS1 = optim.Adam(gS1.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))
optimizerGS2 = optim.Adam(gS2.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))

optimizerDS1 = optim.Adam(dS1.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))
optimizerDS2 = optim.Adam(dS2.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))





def pretrain():
    for epoch in range(opt.niter):
        for i,data in enumerate(dataloader):
            batchsize = data[0].shape[0]

            real64 = data[0].cuda()
            real128 = data[1].cuda()
            condi = data[2].view(batchsize,-1).cuda()
            nzz = torch.randn(batchsize,100).cuda()
            noise = torch.cat((nzz,condi),dim=1)

            label_real = torch.full((batchsize,), real_label).cuda()
            label_fake = torch.full((batchsize,), fake_label).cuda()

            fake64 = gS1(noise)
            fake128 = gS2(fake64.detach())

            ####################################
            #train with DS1
            ####################################
            dS1.zero_grad()
            optimizerDS1.zero_grad()

            DR64 = dS1(real64)
            DF64 = dS2(fake64)
            errD1_real = criterion(DR64, label_real)
            errD1_fake = criterion(DF64, label_fake)
            
            err_D1 = errD1_real.mean().item() + errD1_fake.mean().item()
            
            errD1_real.backward()
            errD1_fake.backward()

            optimizerDS1.step()
            ####################################
            #train with DS2
            ####################################
            dS2.zero_grad()
            optimizerDS2.zero_grad()


            errD2_real = dS2(real128).mean()
            errD2_fake = dS2(fake128.detach()).mean()

            errD2 = 1 - errD2_real + errD2_fake
            
            err_D2 = errD2.mean().item()

            errD2.backward()
            optimizerDS2.step()

            ####################################
            #train with GS1
            ####################################
            gS1.zero_grad()
            optimizerGS1.zero_grad()
        
            # del DF64; del fake64
            fake64 = gS1(noise)
            DF64 = dS1(fake64)
            GFT64 = gT1(noise)
            
            errG1 = criterion(DF64, label_real) +  5 * nn.SmoothL1Loss()(fake64,real64) + 5 * nn.MSELoss()(fake64, GFT64)
            
            err_G1 = errG1.mean().item()
            
            errG1.backward()
            optimizerGS1.step()

            ####################################
            #train with GS2
            ####################################
            gS2.zero_grad()
            optimizerGS2.zero_grad()
            # del DF64; del fake64; del fake128
            fake64 = gS1(noise)
            fake128 = gS2(fake64.detach())

            DF128 = dS2(fake128)
            # del GFT64
            GFT64 = gT1(noise)
            GFT128 = gT2( GFT64 )
            

            errG2 = nn.MSELoss()(fake128, GFT128) + generator_criterion(DF128.mean(), fake128, real128)
            
            err_G2 = errG2.mean().item()
            
            errG2.backward()
            optimizerGS2.step()

            print('[%d/%d][%d/%d] Loss_D1: %.4f Loss_G1: %.4f Loss_D2: %.4f Loss_G2: %.4f'
                  % (epoch, opt.niter, i, len(dataloader), err_D1, err_G1, err_D2, err_G2 ) )
            if i == 0:
                vutils.save_image(real64,
                        '%s/64/real_samples_64.png' % opt.outf,
                        normalize=True)
                vutils.save_image(real128,
                        '%s/128/real_samples_128.png' % opt.outf,
                        normalize=True)
                fake64 = gS1(noise)
                fake128 = gS2(fake64)
                vutils.save_image(fake64.detach(),
                        '%s/64/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                vutils.save_image(fake128.detach(),
                        '%s/128/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
            torch.cuda.empty_cache()
        if(epoch % 10 == 0):
            torch.save(gS1.state_dict(), opt.ckpt+'/netG1_epoch_%d.pth' % (epoch))
            torch.save(dS1.state_dict(), opt.ckpt+'/netD1_epoch_%d.pth' % (epoch))
            torch.save(gS2.state_dict(), opt.ckpt+'/netG2_epoch_%d.pth' % (epoch))
            torch.save(dS2.state_dict(), opt.ckpt+'/netD2_epoch_%d.pth' % (epoch))
            

    torch.save(gS1.state_dict(), opt.ckpt+'/netG1_Final.pth')
    torch.save(dS1.state_dict(), opt.ckpt+'/netD1_Final.pth')
    torch.save(gS2.state_dict(), opt.ckpt+'/netG2_Final.pth')
    torch.save(dS2.state_dict(), opt.ckpt+'/netD2_Final.pth')


if __name__ == '__main__':
    pretrain()