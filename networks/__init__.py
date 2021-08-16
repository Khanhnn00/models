import functools

import torch
import torch.nn as nn
from torch.nn import init

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.normal_(m.weight.data, 0.0, std)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        if classname != "MeanShift":
            print('initializing [%s] ...' % classname)
            init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.Linear)):
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d)):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

####################
# define network
####################

def create_model(opt):
    if opt['mode'] == 'sr':
        net = define_net(opt['networks'])
        return net
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])

# choose one network
def define_net(opt):

    which_model = opt['which_model'].upper()
    print('===> Building network [%s]...'%which_model)

    if which_model == 'DBPN':
        from .dbpn_arch import DBPN
        net = DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                         num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                         upscale_factor=opt['scale'])

    elif which_model == 'D-DBPN':
        from .dbpn_arch import D_DBPN
        net = D_DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                           num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                           upscale_factor=opt['scale'])

    elif which_model == 'D-DBPN_MOD':
        print('D-DPBN_MOD')
        from .dpbn_mod import D_DBPN_MOD
        net = D_DBPN_MOD(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                        num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                        upscale_factor=opt['scale'])
    
    elif which_model == 'D_DBPN_V3':
        print('D_DBPN_V3')
        from .dbpn_v3 import D_DBPN_V3
        net = D_DBPN_V3(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                        num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                        upscale_factor=opt['scale'])

    elif which_model == 'D-DBPN_DN':
        print('D-DPBN_DN')
        from .dbpn_DN import D_DBPN_DN
        net = D_DBPN_DN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                        num_features=opt['num_features'], bp_stages=opt['num_blocks'],
                        upscale_factor=opt['scale'])

    elif which_model == 'RDN_V3':
        from .rdn_v3 import RDN_V3
        net = RDN_V3(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], num_layers = opt['num_layers'],
                             upscale_factor=opt['scale'])

    elif which_model == 'RDN_DN':
        from .rdn_DN import RDN_DN
        net = RDN_DN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], num_layers = opt['num_layers'],
                             upscale_factor=opt['scale'])

    elif which_model == 'RCAN':
        from .rcan_arch import RCAN
        net = RCAN(num_groups=opt['num_groups'],num_blocks=opt['num_blocks'],num_features=opt['num_features'],reduction=opt['reduction'],
                                 scale=opt['scale'],rgb_range=opt['rgb_range'],in_channels=opt['in_channels'],res_scale=opt['res_scale'])    

    elif which_model == 'RCAN_DN':
        from .rcan_DN import RCAN_DN
        net = RCAN_DN(num_groups=opt['num_groups'],num_blocks=opt['num_blocks'],num_features=opt['num_features'],reduction=opt['reduction'],
                                 scale=opt['scale'],rgb_range=opt['rgb_range'],in_channels=opt['in_channels'],res_scale=opt['res_scale'])   
                                 
    elif which_model == 'RCAN_V3':
        from .rcan_v3 import RCAN_V3
        net = RCAN_V3(num_groups=opt['num_groups'],num_blocks=opt['num_blocks'],num_features=opt['num_features'],reduction=opt['reduction'],
                                 scale=opt['scale'],rgb_range=opt['rgb_range'],in_channels=opt['in_channels'],res_scale=opt['res_scale'])  

    elif which_model == 'RCAN_V4':
        from .rcan_v3 import RCAN_V3
        net = RCAN_V3(num_groups=opt['num_groups'],num_blocks=opt['num_blocks'],num_features=opt['num_features'],reduction=opt['reduction'],
                                 scale=opt['scale'],rgb_range=opt['rgb_range'],in_channels=opt['in_channels'],res_scale=opt['res_scale']) 

    elif which_model == 'HAN':
        from .han_arch import HAN
        net = HAN(in_channels=opt['in_channels'],n_resgroups=opt['num_groups'], n_resblocks=opt['num_blocks'],
                                  n_feats=opt['num_features'], reduction=opt['reduction'], rgb_range=opt['rgb_range'],
                                  res_scale=opt['res_scale'])


    elif which_model == 'SRFBN':
        from .srfbn_arch import SRFBN
        net = SRFBN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                  num_features=opt['num_features'], num_steps=opt['num_steps'], num_groups=opt['num_groups'],
                                  upscale_factor=opt['scale'])

    elif which_model.find('GMFN') >= 0:
        from .gmfn_arch import GMFN
        net = GMFN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                               num_features=opt['num_features'], num_steps=opt['num_steps'], num_blocks=opt['num_blocks'],
                               num_reroute_feats=opt['num_reroute_feats'], num_refine_feats=opt['num_refine_feats'],
                               upscale_factor=opt['scale'])

    elif which_model.find('RDN') >= 0:
        from .rdn_arch import RDN
        net = RDN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], num_layers = opt['num_layers'],
                             upscale_factor=opt['scale'])


    elif which_model == 'EDSR':
        # print('EDSR')
        from .edsr_arch import EDSR
        net = EDSR(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'],
                             upscale_factor=opt['scale'])

    elif which_model == 'EDSR_PLUS':
        from .edsr_mod import EDSR_PLUS
        net = EDSR_PLUS(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'],
                             upscale_factor=opt['scale'])

    elif which_model == 'EDSR_DN':
        from .edsr_DN import EDSR_DN
        net = EDSR_DN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'],
                             upscale_factor=opt['scale'])

    elif which_model == 'EDSR_V3':
        from .edsr_v3 import EDSR_V3
        net = EDSR_V3(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'],
                             upscale_factor=opt['scale'])

    elif which_model == 'EDSR_V4':
        from .edsr_v4 import EDSR_V4
        net = EDSR_V4(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], num_blocks = opt['num_blocks'], res_scale=opt['res_scale'],
                             upscale_factor=opt['scale'])

    elif which_model == 'RANDOM':
        from .demo import RANDOM
        net = RANDOM(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                                  num_features=opt['num_features'], num_blocks=opt['num_blocks'], num_groups=opt['num_groups'], 
                                  upscale_factor=opt['scale'])

    else:
        raise NotImplementedError("Network [%s] is not recognized." % which_model)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    return net
