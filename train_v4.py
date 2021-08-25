import argparse, random
from tqdm import tqdm

import torch

import options.options as option
from utils import util
import os
from solvers import create_solver, create_solver_split, create_solver_v2, create_solver_v3, create_solver_abla, create_solver_v4
from data import create_dataloader
from data import create_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    #	parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    #	opt = option.parse(parser.parse_args().opt)
    # opt = option.parse('options/train/train_EDSR_v3.json')
    opt = option.parse('options/train/train_EDSR_v4.json')
    # opt = option.parse('options/train/train_RDN_v3.json')
    # opt = option.parse('options/train/train_RCAN.json')
    # opt = option.parse('options/train/train_RDN.json')
    # opt = option.parse('options/train/train_DBPN_mod.json')
    # opt = option.parse('options/train/train_RDN.json')
    # opt = option.parse('options/train/train_HAN.json')
    # opt = option.parse('options/train/train_EDSR.json')

    # random seed
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)
    print("===> Random Seed: [%d]"%seed)
    random.seed(seed)
    torch.manual_seed(seed)
    loader_list = []

    # create train and val dataloader
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset: %s   Number of images: [%d]' % (train_set.name(), len(train_set)))
            if train_loader is None: raise ValueError("[Error] The training data does not exist")

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            loader_list.append(val_loader)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val_set.name(), len(val_set)))
        
        elif phase == 'val2':
            val2 = create_dataset(dataset_opt)
            val_loader2 = create_dataloader(val2, dataset_opt)
            loader_list.append(val_loader2)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val2.name(), len(val2)))
        
        elif phase == 'val3':
            val3 = create_dataset(dataset_opt)
            val_loader3 = create_dataloader(val3, dataset_opt)
            loader_list.append(val_loader3)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val3.name(), len(val3)))

        elif phase == 'val4':
            val_set4 = create_dataset(dataset_opt)
            val_loader4 = create_dataloader(val_set4, dataset_opt)
            loader_list.append(val_loader4)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val_set4.name(), len(val_set4)))

        elif phase == 'val5':
            val5 = create_dataset(dataset_opt)
            val_loader5 = create_dataloader(val5, dataset_opt)
            loader_list.append(val_loader5)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val5.name(), len(val5)))

        elif phase == 'val6':
            val6 = create_dataset(dataset_opt)
            val_loader6 = create_dataloader(val6, dataset_opt)
            loader_list.append(val_loader6)
            print('===> Val Dataset: %s   Number of images: [%d]' % (val6.name(), len(val6)))

        else:
            raise NotImplementedError("[Error] Dataset phase [%s] in *.json is not recognized." % phase)

    # solver = create_solver_v3(opt)
    # solver = create_solver_split(opt) #for mod
    # solver = create_solver_v2(opt)    #for ablation
    solver = create_solver_v4(opt)
    # solver = create_solver(opt)
    scale = opt['scale']
    model_name = opt['networks']['which_model'].upper()
    print(model_name)

    print('===> Start Train')
    print("==================================================")

    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']


    print("Method: %s || Scale: %d || Epoch Range: (%d ~ %d)"%(model_name, scale, start_epoch, NUM_EPOCH))

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print('\n===> Training Epoch: [%d/%d]...  Learning Rate: %f'%(epoch,
                                                                      NUM_EPOCH,
                                                                      solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        train_loss_list = []
        with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]'%(epoch, NUM_EPOCH), miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                solver.feed_data(batch, is_train=True)
                iter_loss = solver.train_step()
                batch_size = batch['LR'].size(0)
                train_loss_list.append(iter_loss*batch_size)
                t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
                t.update()

        solver_log['records']['train_loss'].append(sum(train_loss_list)/len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print('\nEpoch: [%d/%d]   Avg Train Loss: %.6f' % (epoch,
                                                    NUM_EPOCH,
                                                    sum(train_loss_list)/len(train_set)))
        
        print('===> Validating...',)

        cnt = 1
        epoch_is_best = False
        for sth in loader_list:

            val_loss = []
            val_psnr = []
            val_ssim = []

            for iter, batch in enumerate(sth):
                solver.feed_data(batch)
                iter_loss = solver.test()
                val_loss.append(iter_loss)

                # calculate evaluation metrics
                visuals = solver.get_current_visual()
                psnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale, test_Y=True)
                val_psnr.append(psnr)
                val_ssim.append(ssim)

                if opt["save_image"]:
                    solver.save_current_visual(epoch, iter)

            solver_log['records']['val_loss_{}'.format(cnt)].append(sum(val_loss)/len(val_loss))
            solver_log['records']['psnr_{}'.format(cnt)].append(sum(val_psnr)/len(val_psnr))
            solver_log['records']['ssim_{}'.format(cnt)].append(sum(val_ssim)/len(val_ssim))
            if cnt == 2:
                if solver_log['best_pred'] < sum(val_psnr)/len(val_psnr):
                    solver_log['best_pred'] = sum(val_psnr)/len(val_psnr)
                    epoch_is_best = True
                    solver_log['best_epoch'] = epoch
            cnt +=1 

        # record the best epoch
        # epoch_is_best = False
        # if solver_log['best_pred'] < (sum( solver_log['records']['psnr_2'][epoch-1])/len(solver_log['records']['psnr_2'][epoch-1])):
        #     solver_log['best_pred'] = (sum(solver_log['records']['psnr_2'][epoch-1])/len(solver_log['records']['psnr_2'][epoch-1]))
        #     epoch_is_best = True
        #     solver_log['best_epoch'] = epoch

        # epoch_is_best = False
        # if epoch ==1:
        #     epoch_is_best = True
        #     solver_log['best_pred'] = (sum(train_loss_list)/len(train_set))
        #     solver_log['best_epoch'] = epoch
        # else:
        #     if solver_log['best_pred'] > (sum(train_loss_list)/len(train_set)):
        #         solver_log['best_pred'] = (sum(train_loss_list)/len(train_set))
        #         epoch_is_best = True
        #         solver_log['best_epoch'] = epoch


        # print("[%s] PSNR: %.2f   SSIM: %.4f   Loss: %.6f   Best PSNR: %.2f in Epoch: [%d]" % (val_set.name(),
        #                                                                                       sum(psnr_list_MN)/len(psnr_list_MN),
        #                                                                                       sum(ssim_list_MN)/len(ssim_list_MN),
        #                                                                                       sum(val_loss_list_MN)/len(val_loss_list_MN),
        #                                                                                       solver_log['best_pred'],
        #                                                                                       solver_log['best_epoch']))
                                                                                                                                             
        print("Loss: %.6f   Best loss: %.2f in Epoch: [%d]" % ((sum(train_loss_list)/len(train_set)),
                                                                                              solver_log['best_pred'],
                                                                                              solver_log['best_epoch']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()

        # update lr
        solver.update_learning_rate(epoch)

    print('===> Finished !')


if __name__ == '__main__':
    main()
