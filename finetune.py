# import torch
# import torch.nn as nn
# import os.path as osp
# import os
# import numpy as np
# from PIL import Image
# from torchvision.transforms import ToTensor
# from collections import OrderedDict

# device = torch.device("cpu")

# pretrained = '../SRFBN_x4_BI.pth'
# mine = '../last_ckp.pth'
# from networks.srfbn_arch import SRFBN
# nguoita = SRFBN(in_channels=3, out_channels=3,
#                             num_features=64, num_steps=4, num_groups=6,
#                             upscale_factor=4)
# tao = SRFBN(in_channels=3, out_channels=3,
#                             num_features=64, num_steps=4, num_groups=6,
#                             upscale_factor=4)
# # nguoita = nn.DataParallel(nguoita).cuda()
# # tao = nn.DataParallel(tao).cuda()
# new_state_dict = OrderedDict()
# pretrained = torch.load(pretrained, map_location=device)
# for k, v in pretrained['state_dict'].items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v.cpu()
# nguoita.load_state_dict(new_state_dict)
# mine = torch.load(mine, map_location=device)
# for k, v in mine['state_dict'].items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v.cpu()
# tao.load_state_dict(new_state_dict)
# # print(type(tao))

# pretrained_folder = './visualize_SRFBN_mine'
# if not osp.exists(pretrained_folder):
#     os.mkdir(pretrained_folder)
# min_folder = './visualize_SRFBN_notmine'
# if not osp.exists(min_folder):
#     os.mkdir(min_folder)

# #prepare data
# folder = '../SRbenchmark/LR_x4'

# # print(next(tao.parameters()).is_cuda)
# imgs = os.listdir(folder)
# for img in imgs:
#     name, ext = img.split('.')
#     img = Image.open(osp.join(folder, img))
#     img = np.array(img).astype(np.uint8)
#     img = ToTensor()(img)
#     img = img.to(device)
    
#     img = torch.unsqueeze(img, 0)
#     print(img.is_cuda)
#     print(next(nguoita.parameters()).is_cuda)
#     nguoita_hr = nguoita(img)
#     tao_hr = tao(img)
#     nguoita_hr = nguoita_hr.numpy()
#     tao_hr = tao_hr.numpy()
#     nguoita_hr = Image.fromarray(nguoita_hr.astype(np.uint8))
#     nguoita_hr.save('{}/{}_nguoita.png'.format(pretrained_folder, name))
#     tao_hr = Image.fromarray(tao_hr.astype(np.uint8))
#     tao_hr.save('{}/{}_tao.png'.format(min_folder, name))

import torch
import numpy as np
import torch.nn as nn
import cv2
from scipy import signal
import imageio
from PIL import Image
import os
import os.path as osp
import numbers
import math
from torch.nn import functional as F


'''
convert image to tensor and back
'''
img = imageio.imread(pth, pilmode='RGB') # H W C
np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  #C H W
tensor = torch.from_numpy(np_transpose.copy()).float()  #   to torch.Tensor, so obvious
tensor = tensor.unsqueeze(0)    #1 C H W, must do this if u r going to feed it
tensor.mul_(rgb_range / 255.) #in case you want to scale its range

#K next step I'm gonna make it numpy from torch.Tensor

img = img.squeeze(0).permute(1, 2, 0)   #Back to H W C
img = img.copy().detach().cpu().numpy()
img = Image.from_numpy(img) #Convert to Image to save easier
img.save('path')

'''
Add blur to a tensor using Gaussian distribution
'''

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
        print(kernel.shape)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        print(kernel.shape)

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

kernel = GaussianSmoothing(3, 7, 1.6)   #channel, kernel_size and sigma value
# At this time I assume u alr have an image which is in torch.Tensor form, C W H

tensor = F.pad(tensor, (3, 3, 3, 3), mode='reflect')
'''
The reason u MUST have a padded tensor is because you will want to
keep ur image the original size after convolutional operation
The padding value is depend on your blur kernel size above. I set it 7 so
my padding value would be 3
'''
blur = kernel(tensor) #veri ez

'''
In this step, we are going to add noise to a tensor or even a batch of a lot of tensors
I would use noise generated according to Gaussian distribution
And I will add to a batch of tensors

'''

old_pth_HR = '/storageStudents/K2017/tiepnv/khanhnn/docker/dataset/result/HR_x4'
new_pth_HR = '/storageStudents/K2017/tiepnv/khanhnn/docker/dataset/result/HR_BIDN'
old_pth_LR = '/storageStudents/K2017/tiepnv/khanhnn/docker/dataset/result/LR_x4'
new_pth_HR = '/storageStudents/K2017/tiepnv/khanhnn/docker/dataset/result/LR_BIDN'
def addNoise(pth):
    folder = os.listdir(url)
    names = []
    for name in folder:
        names.append(name)
        img = imageio.imread(osp.join(pth, name), pilmode='RGB')
        img_t = np.ascontiguousarray(img).astype(np.uint8)
        img_t = torch.from_numpy(img_t).permute(2,0,1).unsqueeze(0)
        batch.append(img_t)
    '''
    K now I have a batch of tensors, nxt we will "make some noise go go"
    '''
    noises = np.random.normal(scale=30, size=batch.shape)   #edit this scale
    noises = noises.round()
    noises = torch.from_numpy(noises).short()   #It's better to represent this kernel with int16
    batch = batch.short() + noises  #so do ur image
    batch = torch.clamp(batch, min=0, max=255).type(torch.uint8)    #remember to change it back to uint8
    for i in range(batch.shape[0]):
        img = batch[i].permute(1,2,0).detach().cpu().numpy()
        img = Image.fromarray(img, mode='RGB')
        img.save(osp.join(pth, names[i]))