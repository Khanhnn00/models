import torch
import torch.nn as nn
import os.path as osp
import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from collections import OrderedDict

device = torch.device("cpu")

pretrained = '../SRFBN_x4_BI.pth'
mine = '../last_ckp.pth'
from networks.srfbn_arch import SRFBN
nguoita = SRFBN(in_channels=3, out_channels=3,
                            num_features=64, num_steps=4, num_groups=6,
                            upscale_factor=4)
tao = SRFBN(in_channels=3, out_channels=3,
                            num_features=64, num_steps=4, num_groups=6,
                            upscale_factor=4)
# nguoita = nn.DataParallel(nguoita).cuda()
# tao = nn.DataParallel(tao).cuda()
new_state_dict = OrderedDict()
pretrained = torch.load(pretrained, map_location=device)
for k, v in pretrained['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v.cpu()
nguoita.load_state_dict(new_state_dict)
mine = torch.load(mine, map_location=device)
for k, v in mine['state_dict'].items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v.cpu()
tao.load_state_dict(new_state_dict)
# print(type(tao))

pretrained_folder = './visualize_SRFBN_mine'
if not osp.exists(pretrained_folder):
    os.mkdir(pretrained_folder)
min_folder = './visualize_SRFBN_notmine'
if not osp.exists(min_folder):
    os.mkdir(min_folder)

#prepare data
folder = '../SRbenchmark/LR_x4'

# print(next(tao.parameters()).is_cuda)
imgs = os.listdir(folder)
for img in imgs:
    name, ext = img.split('.')
    img = Image.open(osp.join(folder, img))
    img = np.array(img).astype(np.uint8)
    img = ToTensor()(img)
    img = img.to(device)
    
    img = torch.unsqueeze(img, 0)
    print(img.is_cuda)
    print(next(nguoita.parameters()).is_cuda)
    nguoita_hr = nguoita(img)
    tao_hr = tao(img)
    nguoita_hr = nguoita_hr.numpy()
    tao_hr = tao_hr.numpy()
    nguoita_hr = Image.fromarray(nguoita_hr.astype(np.uint8))
    nguoita_hr.save('{}/{}_nguoita.png'.format(pretrained_folder, name))
    tao_hr = Image.fromarray(tao_hr.astype(np.uint8))
    tao_hr.save('{}/{}_tao.png'.format(min_folder, name))