import torch

import torch.utils.data
import torchvision.utils as vutils


from models.c import Generator

import numpy as np

from models.stage1_teacher import GT1
from models.stage2_teacher import Generator as GT2


from models.stage1_student import GS1
from models.stage2_student import Generator as GS2


import dataload

from torch.autograd import Variable
Tensor = torch.FloatTensor
LongTensor = torch.LongTensor

from collections import OrderedDict

def para(state):
    tmp1 = OrderedDict()
    for k, v in state.items():
        name = k[7:] # remove `module.`
        tmp1[name] = v
    return tmp1

def get_cg():

    cg = Generator()
    cg = torch.load("./ckpts/c.pth")
    return cg

def get_cycle():
    from models.cycle import GeneratorResNet as cycle
    cyclesd = torch.load("./ckpts/cycle.pth")
    cycleg = cycle( (1,256,256), 9)
    cycleg.load_state_dict(cyclesd)
    del cyclesd

    return cycleg


def get_pixg():
    from models.pix import GeneratorUNet as pix
    pixsd = torch.load("./ckpts/pix.pth")
    pixg = pix()
    pixg.load_state_dict(pixsd)
    del pixsd
    torch.cuda.empty_cache()
    return pixg

def get_dcg():
    from models.dc import Generator as dcgan
    dcsd = torch.load("./ckpts/dc.pth")
    dcg = dcgan()
    dcg.load_state_dict(dcsd)
    del dcsd
    torch.cuda.empty_cache()
    return dcg


def get_lsg():
    from models.ls import Generator as ls
    lssd = torch.load("./ckpts/ls.pth")
    lsg = ls()
    lsg.load_state_dict(lssd)
    del lssd
    torch.cuda.empty_cache()
    return lsg

def get_wg():

    wssd = torch.load("./ckpts/w.pth")
    wg = models.w.Generator()
    wg.load_state_dict(wssd.state_dict())
    # del wssd
    # torch.cuda.empty_cache()
    return wg


cg = get_cg().cpu()
cycleg = get_cycle()
pixg = get_pixg()
dcg = get_dcg()
lsg = get_lsg()
import models.w
wg = get_wg()

gt1_sd = torch.load("./ckpts/GT1.pth")
gt2_sd = torch.load("./ckpts/GT2.pth")
gs1_sd = torch.load("./ckpts/GS1.pth")
gs2_sd = torch.load("./ckpts/GS2.pth")

t1 = para(gt1_sd)
t2 = para(gt2_sd)
s1 = para(gs1_sd)
s2 = para(gs2_sd)

gt1 = GT1()
gt2 = GT2(1)
gs1 = GS1()
gs2 = GS2(1)

gt1.load_state_dict(t1)
gt2.load_state_dict(t2)
gs1.load_state_dict(s1)
gs2.load_state_dict(s2)

del t1
del t2
del s1
del s2
torch.cuda.empty_cache()
import time
# time.sleep(5)
import copy
def main():
    dataset = dataload.DataSet()
    DL = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=2, drop_last=True)
    

    for i, imgs in enumerate(DL):
        print(i)
        z = Variable(Tensor(np.random.normal(0, 1, (10, 100))))
        
        real256 = imgs["PD"]
        condi = imgs["T2"]
        cgf = cg(torch.cat((condi.type(torch.FloatTensor),condi.type(torch.FloatTensor)), dim=0))[0].unsqueeze(0)
        cyclef = cycleg(condi.type(Tensor))
        pixf = pixg(condi.type(Tensor))
        dcf = dcg(z)[0].unsqueeze(0)
        lsf = lsg(z)[0].unsqueeze(0)
        wf = wg(z)[0].unsqueeze(0)
        
        LE = gt2(gt1(condi))
        LR = gs2(gs1(condi))

        lb = imgs["T2"]
        GT = imgs["PD"]
        ans = torch.cat( (LE, LR, pixf, cyclef, cgf, dcf, lsf, wf), dim = 0)
        vutils.save_image(ans,
                        '%s%03d.png' % ("./samples/PD/", i),
                        normalize=True)
        # vutils.save_image(LR,
        #                 '%s%03d.png' % ("./samples/", i),
        #                 normalize=True)
        vutils.save_image(lb,
                        '%s%03d.png' % ("./samples/T2/", i),
                        normalize=True)
        vutils.save_image(GT,
                        '%s%03d.png' % ("./samples/GT/", i),
                        normalize=True)
        vutils.save_image(torch.cat((lb,GT),dim=0),
                        '%s%03d.png' % ("./samples/LBGT/", i),
                        normalize=True)
    return


if __name__ == '__main__':
    main()
    