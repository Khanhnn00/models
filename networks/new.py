# Enhanced Deep Residual Networks for Single Image Super-Resolution
# https://arxiv.org/abs/1707.02921

import math
import numbers
import torch
import torch.nn as nn
import numpy as np
import kornia
from kornia import motion_blur
from torch.nn import functional as F

class ImageDIP(nn.Module):
    """
    DIP (Deep Image Prior) for sharp image
    """

    def __init__(self, opt):
        super(ImageDIP, self).__init__()

        input_nc = opt["input_nc"]
        output_nc = opt["output_nc"]

        self.model = skip(
            input_nc,
            output_nc,
            num_channels_down=[128, 128, 128, 128, 128],
            num_channels_up=[128, 128, 128, 128, 128],
            num_channels_skip=[16, 16, 16, 16, 16],
            upsample_mode="bilinear",
            need_sigmoid=True,
            need_bias=True,
            pad=opt["padding_type"],
            act_fun="LeakyReLU",
        )

    def forward(self, img):
        return self.model(img)


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

