from __future__ import generators
import argparse
import os
import random
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor

from lossv6 import GeneratorLoss

#教师网络GT1 GT2
from mdl5.stage1_teacher import GT1
from mdl5.model import Generator as GT2

#学生网络GS1 GS2 DS1 DS2
from mdl5.stage1_student import GS1
from mdl5.stage1_student import DS1
from mdl5.stage2_student import GS2
from mdl5.stage2_student import DS2_c2 as DS2

torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', default=42, type=int, help='manual seed')
parser.add_argument('--card', type=str, default='0',help='select GPU:0 or 1')

parser.add_argument('--outf', default='./result/dis/teachernetv7.4_1216/output', help='folder to output images and model checkpoints')
parser.add_argument('--ckpt', default='./result/dis/teachernetv7.4_1216/ckpt',help='ckpt file path')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

try:
    os.makedirs(opt.outf)
    os.makedirs(opt.ckpt)
    os.makedirs(opt.outf+"/1/")
    os.makedirs(opt.outf+"/2/")
except OSError:
    pass


print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


import dataload
dataset = dataload.DataSet()

dl = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=2)


# 预训练的教师网络
gT1 = GT1().cuda(device="cuda:1")
gT2 = GT2(1).cuda(device="cuda:1")

# 学生网络
gS1 = GS1().cuda(device="cuda:0")
gS2 = GS2(1).cuda(device="cuda:0")

dS1 = DS1().cuda(device="cuda:0")
dS2 = DS2().cuda(device="cuda:0")

from collections import OrderedDict


def parall2cuda(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

g1sd = torch.load("./ckpt_T/netG1_Final.pth")
g2sd = torch.load("./ckpt_T/netG2_Final.pth")

g1 = parall2cuda(g1sd)
g2 = parall2cuda(g2sd)

gT1.load_state_dict(g1)
gT2.load_state_dict(g2)



criterion = nn.MSELoss()
MSEcriterion = nn.MSELoss()
generator_criterion = GeneratorLoss(100, 1, 5).cuda()

real_label = 1.
fake_label = 0.

optimizerGS1 = optim.Adam(gS1.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))
optimizerGS2 = optim.Adam(gS2.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))

optimizerDS1 = optim.Adam(dS1.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))
optimizerDS2 = optim.Adam(dS2.parameters(),lr = 0.0002, betas=(opt.beta1, 0.999))

def pretrain():
    for epoch in range(opt.niter):
        for i,data in enumerate(dl):
            batchsize = data["T2"].shape[0]

            T2 = data["T2"]
            PD = data["PD"]

            batchsize = data["T2"].shape[0]

            real = real1= real2 = PD.cuda(device="cuda:0")
            noise = T2.cuda(device="cuda:0")
            nz1 = noise.cuda(device="cuda:1")

            GFT1 = gT1(nz1).detach()
            GFT2 = gT2(GFT1).detach()

            GFT1_ = GFT1.to(device = "cuda:0")
            GFT2_ = GFT2.to(device = "cuda:0")

            
            valid = Variable(Tensor(batchsize, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batchsize, 1).fill_(0.0), requires_grad=False)

            fake1 = gS1(noise)
            fake2 = gS2(fake1.detach())

            ####################################
            #train with DS1
            ####################################
            dS1.zero_grad()
            optimizerDS1.zero_grad()

            DR1 = dS1(real1).unsqueeze(-1)
            DF1 = dS1(fake1).unsqueeze(-1)
            errD1_real = criterion(DR1, valid)
            errD1_fake = criterion(DF1, fake)
            
            err_D1 = errD1_real.mean().item() + errD1_fake.mean().item()
            
            errD1_real.backward()
            errD1_fake.backward()

            optimizerDS1.step()
            ####################################
            #train with DS2
            ####################################
            dS2.zero_grad()
            fake2 = gS2(fake1.detach())
            
            DR2 = dS2(real).unsqueeze(-1)
            DF2 = dS2(fake2).unsqueeze(-1)

            errD2_real = criterion(DR2, valid)
            errD2_fake = criterion(DF2.detach(), valid)

            #err_D2 = 1 - DR2.mean() + DF2.mean()
            err_D2 = (errD2_real + errD2_fake) * 0.5
            err_D2.backward()
            optimizerDS2.step()

            ####################################
            #train with GS1
            ####################################
            gS1.zero_grad()
            optimizerGS1.zero_grad()
        
            # del DF1; del fake1
            fake1 = gS1(noise)
            DF1 = dS1(fake1)
            
            
            errG1 = criterion(DF1, valid) +  10 * nn.L1Loss()(fake1, GFT1_) + 100 * nn.L1Loss()(fake1, real)
            
            # err_G1 = errG1.mean().item()
            
            errG1.backward()
            optimizerGS1.step()

            ####################################
            #train with GS2
            ####################################
            gS2.zero_grad()
            optimizerGS2.zero_grad()
            # del DF1; del fake1; del fake2
            fake1 = gS1(noise)
            fake2 = gS2(fake1.detach())

            DF2 = dS2(fake2)
            # del GFT1

            errG2 = 10 * nn.L1Loss()(fake2, GFT2_) + generator_criterion(DF2, fake2, real)
            
            errG2.backward()
            optimizerGS2.step()

            print('[%d/%d][%d/%d] Loss_D1: %.4f Loss_G1: %.4f Loss_D2: %.4f Loss_G2: %.4f'
                  % (epoch, opt.niter, i, len(dl), err_D1, errG1.mean().item(), err_D2, errG2.mean().item() ) )
            if i == 0:
                vutils.save_image(real1,
                        '%s/1/real_samples_1.png' % opt.outf,
                        normalize=True)
                vutils.save_image(real2,
                        '%s/2/real_samples_2.png' % opt.outf,
                        normalize=True)
                fake1 = gS1(noise)
                fake2 = gS2(fake1)
                vutils.save_image(fake1.detach(),
                        '%s/1/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                vutils.save_image(fake2.detach(),
                        '%s/2/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
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