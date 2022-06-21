#每个网络单独训练 计算图不联通

from __future__ import print_function
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


from torch.autograd import Variable


from mdl4.stage1_teacher import *
# from mdl3.stage1_teacher import DT1
# from mdl3.stage1_teacher import GT1
from mdl4.model import Generator as GT2
from mdl4.model import Discriminator as DT2


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

dT1 = DT1()
dT2 = DT2()

gT1.apply(weights_init_normal)
gT2.apply(weights_init_normal)
dT1.apply(weights_init_normal)
dT2.apply(weights_init_normal)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)

if(opt.cuda):
    gT1 = nn.DataParallel(gT1).cuda()
    gT2 = nn.DataParallel(gT2).cuda()
    dT1 = nn.DataParallel(dT1).cuda()
    dT2 = nn.DataParallel(dT2).cuda()

criterion = nn.MSELoss()
MSEcriterion = nn.MSELoss()
generator_criterion = GeneratorLoss(100, 1, 5).cuda()
SMTL1criterion = nn.SmoothL1Loss()

# fixed_noise = torch.randn(opt.batchSize, nz, 1, 1).cuda()
real_label = 1.
fake_label = 0.

# setup optimizer
optimizerG1 = optim.Adam(gT1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD1 = optim.Adam(dT1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizerG2 = optim.Adam(gT2.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))
optimizerD2 = optim.Adam(dT2.parameters(), lr=0.0002, betas=(opt.beta1, 0.999))

noise = None

Tensor = torch.cuda.FloatTensor

def pretrain():
    for epoch in range(opt.niter):
        for i, data in enumerate(dl):

            T2 = data["T2"]
            PD = data["PD"]

            batchsize = data["T2"].shape[0]

            real1= real2 = PD.cuda()
            noise = T2.cuda()
            
            valid = Variable(Tensor(batchsize, 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(batchsize, 1).fill_(0.0), requires_grad=False)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            #train with dT1
            ###########################
            dT1.zero_grad()
            fake1 = gT1(noise)

            DR1 = dT1(real1).unsqueeze(-1)
            DF1 = dT1(fake1).unsqueeze(-1)

            D1_x = DR1.mean().item()
            D_G_z1_1 = DF1.mean().item()
            
            errD1_real = criterion(DR1, valid)
            errD1_fake = criterion(DF1.detach(), fake)

            errD1 = (errD1_real + errD1_fake) * 0.5
            errD1.backward()
            optimizerD1.step()
            ###########################
            #train with dT2
            ###########################
            dT2.zero_grad()
            fake2 = gT2(fake1.detach())
            
            DR2 = dT2(real2).unsqueeze(-1)
            DF2 = dT2(fake2).unsqueeze(-1)

            D2_x = DR2.mean().item()
            D_G_z2_2 = DF2.mean().item()

            errD2_real = criterion(DR2, valid)
            errD2_fake = criterion(DF2.detach(), valid)

            #err_D2 = 1 - DR2.mean() + DF2.mean()
            err_D2 = (errD2_real + errD2_fake) * 0.5
            err_D2.backward()
            optimizerD2.step()

            ###########################
            #train with G
            ###########################
            #train with gT1
            ###########################
            gT1.zero_grad()

            DF1 = dT1(fake1).unsqueeze(-1)
            errG1 = 1 * criterion(DF1, valid) + 100 * nn.L1Loss()(fake1,real1)
            errG1.backward()
            optimizerG1.step()
            ###########################
            #train with gT2
            ###########################
            gT2.zero_grad()

            fake1 = gT1(noise)
            fake2 = gT2(fake1.detach())

            DF2 = dT2(fake2)
            errG2 = generator_criterion(DF2, fake2, real2)# + 100 * nn.L1Loss()(fake2, real2)
            errG2.backward()
            optimizerG2.step()


            

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D1(x): %.4f D2(x): %.4f D1(G(z)): (%.4f) D2(G(z)) (%.4f)'
                  % (epoch, opt.niter, i, len(dl),
                     errD1.item(), errG1.item()+errG2.item(), D1_x, D2_x, D_G_z1_1, D_G_z2_2))
            if i % 100 == 0:
                vutils.save_image(real1,
                        '%s/1/real_samples_1.png' % opt.outf,
                        normalize=True)
                vutils.save_image(real2,
                        '%s/2/real_samples_2.png' % opt.outf,
                        normalize=True)
                # fake1 = gT1(noise)
                fake2 = gT2(fake1)
                vutils.save_image(fake1.detach(),
                        '%s/1/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=False)
                vutils.save_image(fake2.detach(),
                        '%s/2/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=False)

        if(epoch % 10 == 0):
            torch.save(gT1.state_dict(), opt.ckpt+'/netG1_epoch_%d.pth' % (epoch))
            torch.save(dT1.state_dict(), opt.ckpt+'/netD1_epoch_%d.pth' % (epoch))
            torch.save(gT2.state_dict(), opt.ckpt+'/netG2_epoch_%d.pth' % (epoch))
            torch.save(dT2.state_dict(), opt.ckpt+'/netD2_epoch_%d.pth' % (epoch))
            

        # do checkpointing
        
    torch.save(gT1.state_dict(), opt.ckpt+'/netG1_Final.pth')
    torch.save(dT1.state_dict(), opt.ckpt+'/netD1_Final.pth')
    torch.save(gT2.state_dict(), opt.ckpt+'/netG2_Final.pth')
    torch.save(dT2.state_dict(), opt.ckpt+'/netD2_Final.pth')

    pass




if __name__ == '__main__':
    pretrain()
