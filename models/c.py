import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.utils as vutils
import dataload as i2d





img_shape = (1, 256, 256)
lb_shape = img_shape


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(int(np.prod(lb_shape)), 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, labels):
        # Concatenate label embedding and image to produce input
        gen_input = labels.view(labels.shape[0],-1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


