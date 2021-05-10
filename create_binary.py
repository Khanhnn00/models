import os
import random
import numpy as np
import scipy.misc as misc
import imageio
from tqdm import tqdm
from PIL import Image
import torch

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
BINARY_EXTENSIONS = ['.npy']
BENCHMARK = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109', 'DIV2K', 'DF2K']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_binary_file(filename):
    return any(filename.endswith(extension) for extension in BINARY_EXTENSIONS)

def _get_paths_from_binary(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    files = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_binary_file(fname):
                binary_path = os.path.join(dirpath, fname)
                files.append(binary_path)
    assert files, '[%s] has no valid binary file' % path
    return files

def _get_paths_from_images(path):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '[%s] has no valid image file' % path
    return images

def get_image_paths(data_type, dataroot):
    paths = None
    if dataroot is not None:
        if data_type == 'img':
            pass
        elif data_type == 'npy':
            if dataroot.find('_npy') < 0 :
                old_dir = dataroot
                dataroot = dataroot + '_npy'
                if not os.path.exists(dataroot):
                    print('===> Creating binary files in [%s]' % dataroot)
                    os.makedirs(dataroot)
                    img_paths = sorted(_get_paths_from_images(old_dir))
                    path_bar = tqdm(img_paths)
                    for v in path_bar:
                        img = imageio.imread(v, pilmode='RGB')
                        ext = os.path.splitext(os.path.basename(v))[-1]
                        name_sep = os.path.basename(v.replace(ext, '.npy'))
                        np.save(os.path.join(dataroot, name_sep), img)
                else:
                    print('===> Binary files already exists in [%s]. Skip binary files generation.' % dataroot)

            paths = sorted(_get_paths_from_binary(dataroot))

        else:
            raise NotImplementedError("[Error] Data_type [%s] is not recognized." % data_type)
    return paths

def create_BI_DN(old_path, new_path):

    print('===> Creating BI and DN files in [%s]' % new_path)
    img_paths = sorted(_get_paths_from_images(old_path))
    print(len(img_paths))
    path_bar = tqdm(img_paths)
    for v in path_bar:
        img = imageio.imread(v, pilmode='RGB')
        img_t = np.ascontiguousarray(img).astype(np.uint8)

        img_bi = Image.fromarray(img_t, mode='RGB')
        ext = os.path.splitext(os.path.basename(v))[-1]
        name_sep = os.path.basename(v.replace(ext, '_BI.png'))
        img_bi.save(os.path.join(new_path, name_sep))

        img_t = torch.from_numpy(img_t).permute(2,0,1).unsqueeze(0)

        noises = np.random.normal(scale=30, size=img_t.shape)   #edit this scale
        noises = noises.round()
        noises = torch.from_numpy(noises).short()   #It's better to represent this kernel with int16

        img_noise = img_t.short() + noises  #so do ur image
        img_noise = torch.clamp(img_noise, min=0, max=255).type(torch.uint8)    #remember to change it back to uint8
        
        img_noise = img_noise.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img_noise = Image.fromarray(img_noise, mode='RGB')
        ext = os.path.splitext(os.path.basename(v))[-1]
        name_sep = os.path.basename(v.replace(ext, '_DN.png'))
        img_noise.save(os.path.join(new_path, name_sep))

# data_type = 'npy'
# # dataroot_HR = '../dataset/Flickr2K/Flickr2K_HR'
# # dataroot_LR = '../dataset/Flickr2K/Flickr2K_LR_bicubic/X4'
# # dataroot_valid = '../dataset/DIV2K/DIV2K_valid_HR'

# # dataroot_HR = './dataset/result/HR_x4_hrx_BD'
# dataroot_LR = './dataset/result/HR_x4_lr_BD'
# # get_image_paths(data_type, dataroot_HR)
# get_image_paths(data_type, dataroot_LR)

old_pth_HR = '../dataset/result/HR_x4'
new_pth_HR = '../dataset/result/HR_BIDN'
old_pth_LR = '../dataset/result/LR_x4'
new_pth_LR = '../dataset/result/LR_BIDN'
create_BI_DN(old_pth_HR, new_pth_HR)
create_BI_DN(old_pth_LR, new_pth_LR)