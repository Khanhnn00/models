import torch.utils.data as data

from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR'])


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR = None, None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 1

        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])
        self.paths_LR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR'])

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                '[Error] HR: [%d] and LR: [%d] have different number of images.'%(
                len(self.paths_LR), len(self.paths_HR))


    def __getitem__(self, idx):
        lr, hr, lr_path, hr_path = self._load_file(idx)
        if self.train:
            lr, lr_x, hr = self._get_patch(lr, hr)
        lr_tensor, lrx_tensor, hr_tensor = common.np2Tensor([lr, lr_x, hr], self.opt['rgb_range'])
        # print(lrx_tensor.shape)
        return {'LR': lr_tensor, 'LR_x': lrx_tensor, 'HR': hr_tensor, 'LR_path': lr_path, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path = self.paths_LR[idx]
        hr_path = self.paths_HR[idx]
        lr = common.read_img(lr_path, self.opt['data_type'])
        hr = common.read_img(hr_path, self.opt['data_type'])

        return lr, hr, lr_path, hr_path


    def _get_patch(self, lr, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr, hr = common.get_patch(
            lr, hr, LR_size, self.scale)
        lr_x = lr.copy()
        lr, lr_x, hr = common.augment([lr, lr_x, hr])
        lr_x = common.add_noise(lr_x, self.opt['noise'])

        return lr, lr_x, hr
