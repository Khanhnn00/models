# Deep Back-Projection Networks for Super-Resolution
# https://arxiv.org/abs/1803.02735

import torch.nn as nn

import networks.blocks as B
import numpy as np
import math
import torch

class DBPN(nn.Module):
    def __init__(self,in_channels, out_channels, num_features, bp_stages, upscale_factor=4, norm_type=None, act_type='prelu'):
        super(DBPN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            projection_filter = 12

        self.feature_extract_1 = B.ConvBlock(in_channels, 128, kernel_size=3, norm_type=norm_type, act_type=act_type)
        self.feature_extract_2 = B.ConvBlock(128, num_features, kernel_size=1, norm_type=norm_type, act_type=act_type)

        bp_units = []
        for _ in range(bp_stages-1):
            bp_units.extend([B.UpprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                                padding=padding, norm_type=norm_type, act_type=act_type),
                            B.DownprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                                  padding=padding, norm_type=norm_type, act_type=act_type)])

        self.bp_units = B.sequential(bp_units)

        self.last_bp_unit = B.UpprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                           padding=padding, norm_type=norm_type, act_type=act_type)
        self.conv_hr = B.ConvBlock(num_features, out_channels, kernel_size=1, norm_type=None, act_type=None)


    def forward(self, x):
        x = self.feature_extract_1(x)
        x = self.feature_extract_2(x)
        x = self.bp_units(x)
        x = self.last_bp_unit(x)
        x = self.conv_hr(x)
        return x

class D_DBPN_MOD(nn.Module):
    def __init__(self,in_channels, out_channels, num_features, bp_stages, upscale_factor=4, norm_type=None, act_type='prelu'):
        super(D_DBPN_MOD, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            projection_filter = 12

        self.feature_extract_1 = B.ConvBlock(in_channels, 256, kernel_size=3, norm_type=norm_type, act_type=act_type)
        self.feature_extract_2 = B.ConvBlock(256, num_features, kernel_size=1, norm_type=norm_type, act_type=act_type)

        bp_units = B.DensebackprojBlock(num_features, num_features, projection_filter, bp_stages, stride=stride, valid_padding=False,
                                                padding=padding, norm_type=norm_type, act_type=act_type)
        self.bp_units = B.sequential(bp_units)
        self.conv_hr = B.ConvBlock(num_features*bp_stages, out_channels, kernel_size=3, norm_type=None, act_type=None)

        # self.network = B.sequential(feature_extract_1, feature_extract_2, bp_units, conv_hr)

    def forward(self, x, is_test=False):
        if is_test == False:
            noises = np.random.normal(scale=30, size=x.shape)
            noises = noises.round()
            ft = torch.from_numpy(noises.copy()).short().cuda()

            x_noise = x.short() + ft.short()
            x_noise = torch.clamp(x_noise, min=0, max=255).type(torch.uint8)

            # ft_x = self.feature_extract_1(x)
            # ft_n = self.feature_extract_1(x_noise.float())
            # ft_x = self.feature_extract_2(ft_x)
            # ft_n = self.feature_extract_2(ft_n)

            # x = ft_x+ft_n

            # x = self.bp_units(x)
            # x = self.conv_hr(x)
            # return x
            
            x = self.feature_extract_1(x_noise.float())
            x = self.feature_extract_2(x)
            x = self.bp_units(x)
            x = self.conv_hr(x)
            return x
        else:
            x = self.feature_extract_1(x)
            x = self.feature_extract_2(x)
            # x = x.mul_(2)
            x = self.bp_units(x)
            x = self.conv_hr(x)
            return x



