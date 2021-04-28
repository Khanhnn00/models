import argparse, time, os
import cv2
import os.path as osp
import options.options as option
from utils import util
from networks import create_model
# import torch
# import torch.nn as nn


def main():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    model = create_model(opt)

    checkpoint = torch.load(opt['solver']['pretrained_path'])

    model.module.load_state_dict(checkpoint['state_dict'])
    # create solver (and load model)

    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))

    inps = './inputs'
    save_dir = './outputs'
    if not osp.exists(save_dir):
        os.mkdir('./outputs')

    imgs = os.listdir(inps)
    
    for tmp in imgs:
        img = osp.join(inps, tmp)
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.Tensor(img)
        
        output = model.forward(img)
        
        cv2.imwrite(osp.join(save_dir, tmp), output)

    print("==================================================")
    print("===> Finished !")

if __name__ == '__main__':
    main()