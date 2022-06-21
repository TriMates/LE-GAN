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


import torchio

# from models0.stage1_teacher import DT1#FOR DRIVE
# from models0.stage1_teacher import GT1#FOR DRIVE
from model import Generator as GT2
from model import Discriminator as DT2

from models1.stage1_teacher import DT1#FOR ISIC
from models1.stage1_teacher import GT1_c16 as GT1#FOR ISIC


# from model.stage1_student import GS1,DS1
# from model.stage2_student import GS2,DS2

from loss import GeneratorLoss
torch.autograd.set_detect_anomaly(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./result/pre/teachernetv1_1128/output', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--ckpt', default='./result/pre/teachernetv1_1128/ckpt',help='ckpt file path')
parser.add_argument('--card', type=str, default='0',help='select GPU:0 or 1')

opt = parser.parse_args()

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


if opt.manualSeed is None:
    opt.manualSeed = 42
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  
# if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
#     raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)


import dataloader
dataset = dataloader.DatasetLoaders()

dl = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=2)


nz = int(opt.nz)


# custom weights initialization called on netG and netD



gT1 = GT1()
gT2 = GT2(1)

dT1 = DT1()
dT2 = DT2()

# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)

if(opt.cuda):
    gT1 = nn.DataParallel(gT1).cuda()
    gT2 = nn.DataParallel(gT2).cuda()
    dT1 = nn.DataParallel(dT1).cuda()
    dT2 = nn.DataParallel(dT2).cuda()

criterion = nn.BCELoss()
MSEcriterion = nn.MSELoss()
generator_criterion = GeneratorLoss().cuda()
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

def pretrain():
    for epoch in range(opt.niter):
        for i, data in enumerate(dl):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with D real
            T2 = data["T2"].unsqueeze(1)
            PD = data["PD"].unsqueeze(1)

            batchsize = data["T2"].shape[0]

            real1= real2 = PD.cuda()
            noise = T2.cuda()
            label = torch.full((batchsize,), real_label).cuda()

            DR1 = dT1(real1)
            DR2 = dT2(real2)
            
            dT1.zero_grad()
            dT2.zero_grad()
            errD1_real = criterion(DR1, label)
            errD1_real.backward()

            D1_x = DR1.mean().item()
            D2_x = DR2.mean().item()
            fake1 = gT1(noise)
            fake2 = gT2(fake1)
            label.fill_(fake_label)
            DF1 = dT1(fake1.clone().detach())
            DF2 = dT2(fake2.clone().detach())

            errD1_fake = criterion(DF1,label)
            errD1_fake.backward()
            D_G_z1_1 = DF1.mean().item()
            D_G_z1_2 = DF2.mean().item()
            
            errD = errD1_real + errD1_fake
            optimizerD1.step()
            
            # train with DT2
            real_out = dT2(real2.detach()).mean()
            fake_out = dT2(fake2.detach()).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward()
            optimizerD2.step()
            ###########################
            #train with G
            ###########################
            gT1.zero_grad()
            gT2.zero_grad()
            label.fill_(real_label)

            output1 = dT1(fake1)
            errG1 = 1 * criterion(output1, label) + 5 * nn.SmoothL1Loss()(fake1,real1)
            errG1.backward(retain_graph=True)

            
            fake_out = dT2(fake2.detach()).mean()
            errG2 = generator_criterion(fake_out, fake2, real2)
            errG2.backward()

            optimizerG1.step()
            optimizerG2.step()

            D_G_z2_1 = output1.mean().item()
            D_G_z2_2 = fake_out.item()
            

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D1(x): %.4f D2(x): %.4f D1(G(z)): (%.4f,%.4f) / (%.4f,%.4f)'
                  % (epoch, opt.niter, i, len(dl),
                     errD.item(), errG1.item()+errG2.item(), D1_x,D2_x, D_G_z1_1,D_G_z1_2, D_G_z2_1,D_G_z2_2))
            if i % 100 == 0:
                vutils.save_image(real1,
                        '%s/1/real_samples_1.png' % opt.outf,
                        normalize=True)
                vutils.save_image(real2,
                        '%s/2/real_samples_2.png' % opt.outf,
                        normalize=True)
                fake1 = gT1(noise)
                fake2 = gT2(fake1)
                vutils.save_image(fake1.detach(),
                        '%s/1/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                vutils.save_image(fake2.detach(),
                        '%s/2/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                        

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
