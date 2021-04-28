import os
import random
import numpy as np
import scipy.misc as misc
import imageio
from tqdm import tqdm
import cv2
from PIL import Image
import os.path as osp

import torch

def createBDImgs(dataroot):

    old_dir = dataroot
    dataroot = old_dir + '_hr_BD'
    dataroot_LR = old_dir + '_lr_BD'
    dataroot_HRx = old_dir + '_hrx_BD'
    if not os.path.exists(dataroot_LR) and not os.path.exists(dataroot_HRx):
        os.makedirs(dataroot_LR)
        os.makedirs(dataroot_HRx)
    elif os.path.exists(dataroot_LR) and os.path.exists(dataroot_HRx):
        if len(os.listdir(dataroot_LR))> 0 and len(os.listdir(dataroot_HRx)) > 0:
            print('Some folders are already exist')
            return

    print('===> Creating binary files in [%s]' % dataroot)
    img_paths = sorted(os.listdir(old_dir))
    path_bar = tqdm(img_paths)
    for v in path_bar:
        hr = Image.open(osp.join(old_dir,v)) #PIL file
        hr = np.array(hr).astype(np.uint8)
        hr_bd_cv2 = cv2.GaussianBlur(hr,(7,7),1.6) 
        lr_bd = misc.imresize(hr_bd_cv2, 1 / 4, interp='bicubic') #np array, range 0-255, w h c
        name_lr = v.split('.')[0] + '_x4' + '.png'
        name_hrx = v.split('.')[0] + '_hrx' + '.png'

        lr_bd = Image.fromarray(lr_bd)
        hr_bd_cv2 = Image.fromarray(hr_bd_cv2)
        lr_bd.save(os.path.join(dataroot_LR, name_lr))
        hr_bd_cv2.save(os.path.join(dataroot_HRx, name_hrx))


        

createBDImgs('./dataset/result/HR_x4')






# path = '../dataset/DIV2K/DIV2K_train_HR/0001.png'   #path to img

# hr = Image.open(path) #PIL file
# hr = np.array(hr).astype(np.uint8) #np array, range 0-255, w h c

# #create bd: hr -> add gaussian blur -> downsample
# hr_bd_cv2 = cv2.GaussianBlur(hr,(7,7),1.6)  #cv2
# blur_np = np.random.normal(0, 1.6, hr.shape)
# hr_bd_np = hr.astype(np.int16) + blur_np.astype(np.int16)
# hr_bd_np = hr_bd_np.clip(0, 255).astype(np.uint8)
# lr_bd = misc.imresize(hr_bd_cv2, 1 / 4, interp='bicubic')

# hr_bd_cv2 = Image.fromarray(hr_bd_cv2.astype(np.uint8))
# hr_bd_cv2.save('hr_bd_cv2.png')
# # hr_bd_np = Image.fromarray(hr_bd_np.astype(np.uint8))
# # hr_bd_np.save('hr_bd_np.png')
# lr_bd = Image.fromarray(lr_bd.astype(np.uint8))
# lr_bd.save('lr_bd.png')
# #create dn: hr -> downsample -> add gaussian noise level 30
# lr_dn = misc.imresize(hr, 1 / 4, interp='bicubic')
# noises = np.random.normal(scale=30, size=lr_dn.shape)
# noises = noises.round()
# lr_dn = lr_dn.astype(np.int16) + noises.astype(np.int16)
# lr_dn = lr_dn.clip(0, 255).astype(np.uint8)
# noises = np.random.normal(scale=30, size=hr.shape)
# noises = noises.round()
# hr_dn = hr.astype(np.int16) + noises.astype(np.int16)
# hr_dn = hr_dn.clip(0, 255).astype(np.uint8)

# hr_dn = Image.fromarray(hr_dn.astype(np.uint8))
# hr_dn.save('hr_dn.png')
# lr_dn = Image.fromarray(lr_dn.astype(np.uint8))
# lr_dn.save('lr_dn.png')
# hr = Image.fromarray(hr.astype(np.uint8))
# hr.save('hr.png')
# lr = misc.imresize(hr, 1 / 4, interp='bicubic')
# lr = Image.fromarray(lr.astype(np.uint8))
# lr.save('lr.png')