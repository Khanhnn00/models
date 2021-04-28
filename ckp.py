import argparse, random
from tqdm import tqdm

import torch
import piq
import os
import numpy as np
import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
from networks import create_model
import cv2
# ssim = piq.ssim().cuda()
def main():
    loss = piq.SSIMLoss(data_range=255.).cuda()
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')

    # opt = option.parse('options/train/train_SRFBN_BD.json')
    # opt = option.parse('options/train/train_SRFBN_DN.json')
    #opt = option.parse('options/train/train_SRFBN_BI.json')
    opt = option.parse('options/train/train_EDSR.json')

    # val_set = create_dataset(opt['datasets']['val'])
    # val_loader = create_dataloader(val_set, opt['datasets']['val'])
    # print('===> Val Dataset: %s   Number of images: [%d]' % (val_set.name(), len(val_set)))

    model = create_model(opt)
    ckp = torch.load('../experiments/old_exp/EDSR_in3f256_x4/epochs/last_ckp.pth')
    print(ckp['epoch'])
    print(ckp.keys())
    model.module.load_state_dict(ckp['state_dict'])
    # print(model)
    model.eval()
    

    # scale = opt['scale']
    # model_name = opt['networks']['which_model'].upper()
    with torch.no_grad():
        
        # for iter, batch in enumerate(val_loader):
        #     # print(iter)
            
        #     input = batch['LR']
        #     print(input.shape)
        #     outputs = model(input)
        #     count = 0
        #     for pred in outputs:
        #         # print(pred.shape)
        #         pred = pred.clamp(min=0, max=255).round().cuda()
        #         print(pred.min())
                
        #         this_loss = loss(pred, batch['HR'].cuda()).cuda()
        #         print(this_loss)
        #         pred = torch.squeeze(pred)
        #         # print(pred.shape)
        #         pred = pred.cpu().numpy()
        #         hr = batch['HR'].cpu().numpy()
        #         pred = np.transpose(pred, (1,2,0))
        #         pred = np.transpose(pred, (1,2,0))
        #         cv2.imwrite('{}_{}.jpg'.format(iter, count), pred)
        #         count+=1
        img = cv2.imread('./SRbenchmark/valid_div_x4/0804x4.png')
        hr = cv2.imread('./SRbenchmark/valid_div/0804.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        hr = np.transpose(hr, (2, 0, 1))
        img = torch.from_numpy(img)
        hr = torch.from_numpy(hr)
        img = img.unsqueeze(0).cuda()
        print(img.shape)
        hr = hr.unsqueeze(0).cuda()
        pred = model(img.float())
        pred = pred.clamp(min=0, max=255).round().cuda()
        print(pred.min())
        
        this_loss = loss(pred, hr.cuda()).cuda()
        print(this_loss)
        pred = torch.squeeze(pred)
        # print(pred.shape)
        pred = pred.cpu().numpy()
        hr = batch['HR'].cpu().numpy()
        pred = np.transpose(pred, (1,2,0))
        # pred = np.transpose(pred, (1,2,0))
        cv2.imwrite('{}_{}.jpg'.format(iter, count), pred)

    print('===> Finished !')


if __name__ == '__main__':
    main()
