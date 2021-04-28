import torch
import torch.nn as nn
from torch.nn import functional as F
import numbers
import math
from .blocks import ConvBlock, DeconvBlock, MeanShift

class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # print(kernel.shape)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        # print(kernel.shape)

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)

class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()

        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        # print(x.shape)
        x = self.compress_in(x)

        lr_features = []
        hr_features = []
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True
        
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(inplace=False), res_scale=0.1):

        super(ResBlock, self).__init__()
        
        self.body = nn.ModuleList()
        for i in range(2):
            m = []
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, mode='C'))
            m.append(nn.BatchNorm2d(n_feats))
            m.append(act)
            self.body.append(nn.Sequential(*m))

        # self.res_scale = res_scale

    def forward(self, x):
        out1 = self.body[0](x)
        out2 = self.body[1](x+out1)
        res = x + out2

        return res
        
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class RANDOM(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, num_blocks, num_groups, upscale_factor, act_type = 'prelu', norm_type = None):
        super(RANDOM, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_blocks = num_blocks
        self.num_features = num_features
        self.upscale_factor = upscale_factor
        #self.blur_matrix = gaussian blur kernel

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_mean, rgb_std)

        #extract in
        self.conv_in = ConvBlock(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)
        self.add_blur = GaussianSmoothing(3, 7, 1.6)
        self.conv_in_blur = ConvBlock(in_channels, 4*num_features,
                                 kernel_size=3,
                                 act_type=act_type, norm_type=norm_type)
        self.feat_in_blur = ConvBlock(4*num_features, num_features,
                                 kernel_size=1,
                                 act_type=act_type, norm_type=norm_type)
                              
        # BI feature extraction
        self.res_blocks = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for _ in range(self.num_blocks-1):
            self.res_blocks.append(ResBlock(ConvBlock, num_features, kernel_size=3))
            self.blocks.append(FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type))

        # basic block
        self.blocks.append(FeedbackBlock(num_features, num_groups, upscale_factor, act_type, norm_type))

        # reconstruction block
    	# uncomment for pytorch 0.4.0
        # self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='bilinear')

        self.out = ConvBlock(num_features, num_features,
                               kernel_size=1, stride=1,
                               act_type='prelu', norm_type=norm_type)
        self.upsampler = Upsampler(ConvBlock, scale=4, n_feats=64)
      
        self.conv_out = ConvBlock(num_features, out_channels,
                                  kernel_size=3,
                                  act_type=None, norm_type=norm_type)

        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x, is_test=False):
        self._reset_state()

        x = self.sub_mean(x)
		# uncomment for pytorch 0.4.0
        # inter_res = self.upsample(x)
		
		# comment for pytorch 0.4.0
        inter_res = nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        if is_test == False:
            x_blur = F.pad(x, (3, 3, 3, 3), mode='reflect')
            x = self.conv_in(x)
            feat_x = self.feat_in(x)
            x_blur = self.add_blur(x_blur)
            # print(x_blur.shape)
            x_blur = self.conv_in(x_blur)
            feat_blur = self.feat_in(x_blur)
            
            feat_mid = torch.add(feat_x, feat_blur)
            
            feat_mid = self.blocks[0](feat_mid)
            
            for _ in range(self.num_blocks-1):
                feat_x = self.res_blocks[_](feat_x)
                # print('feat_x: {}').format(feat_x.shape)
                feat_blur = self.res_blocks[_](feat_blur)
                # print('feat_blur: {}').format(feat_blur.shape)
                feat_mid = self.blocks[_+1](torch.add(torch.add(feat_x,feat_blur),feat_mid))
                # h = torch.add(inter_res, self.conv_out(self.out(h)))
                # h = self.add_mean(h)
                # outs.append(h)
            feat_mid = self.out(feat_mid)
            feat_mid = self.upsampler(feat_mid)
            feat_mid = self.conv_out(feat_mid)
            feat_mid = torch.add(feat_mid, inter_res)
            h = self.add_mean(feat_mid)
            return h # return output of every timesteps
        else:
            x = self.conv_in(x)
            feat_x = self.feat_in(x)
            feat_mid = self.blocks[0](feat_x)
            for _ in range(self.num_blocks -1):
                feat_x = self.res_blocks[_](feat_x)
                feat_mid = self.blocks[_+1](torch.add(feat_x,feat_mid))
                # h = torch.add(inter_res, self.conv_out(self.out(h)))
                # h = self.add_mean(h)
                # outs.append(h)
            feat_mid = self.out(feat_mid)
            feat_mid = self.upsampler(feat_mid)
            feat_mid = self.conv_out(feat_mid)
            feat_mid = torch.add(feat_mid, inter_res)
            h = self.add_mean(feat_mid)
            return h # return output of every timesteps
        

    def _reset_state(self):
        [x.reset_state() for x in self.blocks]