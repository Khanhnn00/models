from glob import glob
from flags import *
import os
import os.path as osp
from scipy import misc
import numpy as np
import datetime
import imageio
import torch
from multiprocessing.dummy import Pool as ThreadPool

dataroot_HR = '../../dataset/Flickr2K/Flickr2K_HR'
dataroot_LR = '../../dataset/Flickr2K/Flickr2K_LR_bicubic/X4'

save_dir_HR = 'HR_x4'
save_dir_LR = 'LR_x4'

save_HR = osp.join(dataroot_HR, save_dir_HR)
save_LR = osp.join(dataroot_LR, save_dir_LR)
print(save_HR, save_LR)
hr_imgs = sorted(glob(os.path.join(dataroot_HR, '*.png')))
lr_imgs = sorted(glob(os.path.join(dataroot_LR, '*.png')))

# print(len(hr_imgs), len(lr_imgs))

def randomHorizontalFlip(hr_img, lr_img, hr_name, lr_name):
    misc.imsave(save_HR + '/' + hr_name, hr_img)
    misc.imsave(save_LR + '/' + lr_name, lr_img)
    if torch.uniform() < 0.5:
        rot180_img = misc.imrotate(hr_img, 180)
        rot180_img_x4 = misc.imrotate(lr_img, 180)
        misc.imsave(save_HR + '/' + hr_name.split('.')[0] + '_rot180' + hr_name.split('.')[1], rot180_img)
        misc.imsave(save_LR + '/' + lr_name.split('.')[0] + '_rot180' + lr_name.split('.')[1], rot180_img_x4)
    rot90_img = misc.imrotate(hr_img, 90)
    rot90_img_x4 = misc.imrotate(lr_img, 90)

    misc.imsave(save_HR + '/' + hr_name.split('.')[0] + '_rot90' + hr_name.split('.')[1], rot90_img)
    misc.imsave(save_LR + '/' + lr_name.split('.')[0] + '_rot90' + lr_name.split('.')[1], rot90_img_x4)

def getRGBmean():
    R_mean = 0
    G_mean = 0
    B_mean = 0
    for i in range(len(hr_imgs)):
        im = imageio.imread(hr_imgs[i], pilmode="RGB")
        h,w,c = im.shape
        tmp_r = np.sum(im[:,:,0])
        # print(tmp_r)
        tmp_g = np.sum(im[:,:,1])
        tmp_b = np.sum(im[:,:,2])
        R_mean += tmp_r/(h*w)
        G_mean += tmp_g/(h*w)
        B_mean += tmp_b/(h*w)
    R_mean = R_mean/len(hr_imgs)
    G_mean = G_mean/len(hr_imgs)
    B_mean = B_mean/len(hr_imgs)
    return R_mean, G_mean, B_mean

# print(getRGBmean())
print(hr_imgs[1])


    
