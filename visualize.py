import os
import random
import numpy as np
import scipy.misc as misc
import imageio
from tqdm import tqdm
import cv2
from PIL import Image

import torch

path = '../dataset/DIV2K/DIV2K_train_HR/0001.png'   #path to img

hr = Image.open(path) #PIL file
hr = np.array(hr).astype(np.uint8) #np array, range 0-255, w h c

#create bd: hr -> add gaussian blur -> downsample
hr_bd_cv2 = cv2.GaussianBlur(hr,(7,7),1.6)  #cv2
blur_np = np.random.normal(0, 1.6, hr.shape)
hr_bd_np = hr.astype(np.int16) + blur_np.astype(np.int16)
hr_bd_np = hr_bd_np.clip(0, 255).astype(np.uint8)
lr_bd = misc.imresize(hr_bd_cv2, 1 / 4, interp='bicubic')

hr_bd_cv2 = Image.fromarray(hr_bd_cv2.astype(np.uint8))
hr_bd_cv2.save('hr_bd_cv2.png')
# hr_bd_np = Image.fromarray(hr_bd_np.astype(np.uint8))
# hr_bd_np.save('hr_bd_np.png')
lr_bd = Image.fromarray(lr_bd.astype(np.uint8))
lr_bd.save('lr_bd.png')
#create dn: hr -> downsample -> add gaussian noise level 30
lr_dn = misc.imresize(hr, 1 / 4, interp='bicubic')
noises = np.random.normal(scale=30, size=lr_dn.shape)
noises = noises.round()
lr_dn = lr_dn.astype(np.int16) + noises.astype(np.int16)
lr_dn = lr_dn.clip(0, 255).astype(np.uint8)
noises = np.random.normal(scale=30, size=hr.shape)
noises = noises.round()
hr_dn = hr.astype(np.int16) + noises.astype(np.int16)
hr_dn = hr_dn.clip(0, 255).astype(np.uint8)

hr_dn = Image.fromarray(hr_dn.astype(np.uint8))
hr_dn.save('hr_dn.png')
lr_dn = Image.fromarray(lr_dn.astype(np.uint8))
lr_dn.save('lr_dn.png')
hr = Image.fromarray(hr.astype(np.uint8))
hr.save('hr.png')
lr = misc.imresize(hr, 1 / 4, interp='bicubic')
lr = Image.fromarray(lr.astype(np.uint8))
lr.save('lr.png')