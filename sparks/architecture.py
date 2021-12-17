import sys
import os
import logging
import argparse
import configparser
import glob

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import wandb

import unet
from dataset_tools import random_flip, compute_class_weights_puffs, weights_init
from datasets import SparkDataset, SparkTestDataset
from training_tools import training_step, test_function_fixed_t, sampler
from metrics_tools import take_closest
from focal_losses import FocalLoss


class TempRedUNet(nn.Module):
    def __init__(self, unet_config):
        super().__init__()

        padding = {'valid': 0, 'same': unet_config.dilation}[unet_config.border_mode]

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=1,
                               kernel_size=(3,3,3),
                               dilation=1, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=1,
                               kernel_size=(3,1,1), stride=(2,1,1),
                               dilation=1, padding=(padding,0,0))
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=4,
                               kernel_size=(3,1,1), stride=(2,1,1),
                               dilation=1, padding=(padding,0,0))

        self.unet = unet.UNetClassifier(unet_config)




    def forward(self, x):
        #print("input shape", x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        #print("unet input shape", x.shape)
        x = self.unet(x)
        #print("output shape", x.shape)
        return x
