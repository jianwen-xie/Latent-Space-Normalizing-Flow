from binascii import Incomplete
import sys
import os

import argparse
import json
import random
import shutil
import copy
import logging
import datetime
import pickle
import itertools
import time
import math
from math import log, pi, exp
import numpy as np
from tqdm import tqdm
from scipy import io as sio
# I do not know why it must load before "import torchvision"
# from fid_v2_tf_cpu import fid_score

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import pygrid
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import _netG, _netF, weights_init_xavier, _netG_celeba, _netG_mnist, _netG_cifar10
import pytorch_fid_wrapper as pfw

##########################################################################################################
## Parameters

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train", help='training or test mode')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--abnormal', type=int, default=-1, help='training or test mode')
    parser.add_argument('--load_checkpoint', type=str, default="", help='load checkpoint')
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--device', type=int, default=0, help='training or test mode')
    parser.add_argument('--output_dir', type=str, default="default", help='training or test mode')
    parser.add_argument('--dataset', type=str, default='celeba_crop', choices=['svhn', 'celeba', 'celeba_crop', 'mnist', 'mnist_ad', 'cifar10'])
    parser.add_argument('--incomplete_train', type=str, default=None, help='training or test mode')
    parser.add_argument('--data_size', type=int, default=1000000)
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ngf',type=int,  default=128, help='feature dimensions of generator')

    parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='prior of factor analysis')
    parser.add_argument('--g_activation', type=str, default='lrelu')
    parser.add_argument('--g_activation_leak', type=float, default=0.2)
    parser.add_argument('--g_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of langevin') # 0.1
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--g_batchnorm', default=False, type=bool, help='batch norm')

    parser.add_argument('--f_n_levels', default=1, type=int, help='')
    parser.add_argument('--f_depth', default=5, type=int, help='') # 10
    parser.add_argument('--f_flow_permutation', default=2, type=int, help='')
    parser.add_argument('--f_width', default=64, type=int, help='')
    parser.add_argument('--f_flow_coupling', default=1, type=int, help='')

    parser.add_argument('--g_lr', default=0.0003, type=float) # 0.0004
    parser.add_argument('--f_lr', default=0.0003, type=float) # 0.0004

    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--f_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')

    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--f_max_norm', type=float, default=100, help='max norm allowed')

    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')
    parser.add_argument('--f_decay', default=0, help='weight decay for flow')

    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for gen')
    parser.add_argument('--f_gamma', type=float, default=0.998, help='lr decay for flow')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--f_beta1', default=0.5, type=float)
    parser.add_argument('--f_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=20, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')

    parser.add_argument('--n_ckpt', type=int, default=1, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')    #
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')
    parser.add_argument('--n_fid_samples', type=int, default=50000)


    return parser.parse_args()

##########################################################################################################
## Data

class IncompleteDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, masks, data_size=1000000):
        self.dataset = dataset 
        self.data_size = min(data_size, len(self.dataset))
        self.masks = self.generate_mask(masks) if type(masks) is str else masks 
        if self.masks.shape[0] < self.data_size: 
            print("Warning! masks is smaller than data_size.")
            self.data_size = self.masks.shape[0]
    def __len__(self):
        return self.data_size
    def __getitem__(self, idx): 
        x, y = self.dataset[idx]
        return x, y, torch.from_numpy(self.masks[idx]).to(x.device)
    def generate_mask(self, mask_type):
        patchs = np.load("data/masks_gt.npz")
        ratio = {10: 7, 20: 15, 30: 24, 40: 35, 50: 48, 55: 56, 60: 65, 65: 76, 70: 90, 75: 108, 80: 134, 85: 176, 90: 250}
        if mask_type[:4] == "salt":
            num_patch = ratio[int(mask_type[5:])]
            patch_size = 8
            rad1 = patchs["salt_pepper_8"]
            masks = np.ones((self.data_size, 3, 64, 64), dtype=bool)
            for i in range(self.data_size): 
                for j in range(num_patch): 
                    masks[i, :, rad1[i, j, 0]:rad1[i, j, 0]+patch_size, rad1[i, j, 1]:rad1[i, j, 1]+patch_size] = 0
            print("Use salt %d patch: %.8f/100 is covered." % (num_patch, 1 - masks.mean()))
        elif mask_type[:4] == "sing": 
            patch_size = int(mask_type[5:])
            masks = np.ones((self.data_size, 3, 64, 64), dtype=bool)
            rad1 = patchs["single_mask_size_%d" % patch_size]
            for i in range(self.data_size): 
                masks[i, :, rad1[i, 0]:rad1[i, 0]+patch_size, rad1[i, 1]:rad1[i, 1]+patch_size] = 0
            print("Use single mask size %d" % patch_size)
        elif mask_type == "ot_mask": 
            return sio.loadmat('./data/celebA_masks_10000_3_50.mat')['masks']
        else:
            raise NotImplementedError
        return masks

def get_dataset(args):

    fs_prefix = './'

    if args.dataset == 'mnist':
        import torchvision.transforms as transforms
        ds_train = torchvision.datasets.MNIST(fs_prefix + 'data/{}'.format(args.dataset), download=True,
                                             transform=transforms.Compose([
                                             transforms.Resize(args.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.5, 0.5),
                               ]))
        if args.abnormal != -1: 
            Y = [y for (x, y) in ds_train]
            selection = [i for i, y in enumerate(Y) if y != args.abnormal]
            ds_train = torch.utils.data.Subset(ds_train, selection)
        ds_val = torchvision.datasets.MNIST(fs_prefix + 'data/{}'.format(args.dataset), download=True, train=False,
                                             transform=transforms.Compose([
                                             transforms.Resize(args.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.5, 0.5),
                               ]))
        if args.data_size < 10000:
            ds_train = torch.utils.data.Subset(ds_train, np.arange(args.data_size))
        return ds_train, ds_val

    if args.dataset == "mnist_ad": 

        assert args.abnormal != -1 
        data = np.load('data/mnist.npz')

        full_x_data = np.concatenate([data['x_train'], data['x_test'], data['x_valid']], axis=0).reshape(-1, 1, 28, 28) * 2 - 1
        full_y_data = np.concatenate([data['y_train'], data['y_test'], data['y_valid']], axis=0)

        normal_x_data = full_x_data[full_y_data!= args.abnormal]
        normal_y_data = full_y_data[full_y_data!= args.abnormal]

        inds = np.random.permutation(normal_x_data.shape[0])
        normal_x_data = normal_x_data[inds]
        normal_y_data = normal_y_data[inds]

        index = int(normal_x_data.shape[0]*0.8)

        training_x_data = normal_x_data[:min(index, args.data_size)]
        training_y_data = normal_y_data[:min(index, args.data_size)]

        testing_x_data = np.concatenate([normal_x_data[index:], full_x_data[full_y_data == args.abnormal]], axis=0)
        testing_y_data = np.concatenate([normal_y_data[index:], full_y_data[full_y_data == args.abnormal]], axis=0)
        inds = np.random.permutation(testing_x_data.shape[0])
        testing_x_data = testing_x_data[inds]
        testing_y_data = testing_y_data[inds]

        ds_train = torch.utils.data.TensorDataset(torch.Tensor(training_x_data), torch.Tensor(training_y_data))
        ds_val = torch.utils.data.TensorDataset(torch.Tensor(testing_x_data), torch.Tensor(testing_y_data))
        return ds_train, ds_val

    if args.dataset == 'svhn':
        import torchvision.transforms as transforms
        ds_train = torchvision.datasets.SVHN(fs_prefix + 'data/{}'.format(args.dataset), download=True,
                                             transform=transforms.Compose([
                                             transforms.Resize(args.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        ds_val = torchvision.datasets.SVHN(fs_prefix + 'data/{}'.format(args.dataset), download=True, split='test',
                                             transform=transforms.Compose([
                                             transforms.Resize(args.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        return ds_train, ds_val

    if args.dataset == 'celeba':

        import torchvision.transforms as transforms

        ds_train = torchvision.datasets.CelebA(fs_prefix + 'data', split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(args.img_size),
                                                        transforms.CenterCrop(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        ds_val = torchvision.datasets.CelebA(fs_prefix + 'data', split='valid', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(args.img_size),
                                                        transforms.CenterCrop(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return ds_train, ds_val

    if args.dataset == 'celeba_crop':

        crop = lambda x: transforms.functional.crop(x, 45, 25, 173-45, 153-25)

        import torchvision.transforms as transforms

        ds_train = torchvision.datasets.CelebA(fs_prefix + 'data', split='train', download=False,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        ds_val = torchvision.datasets.CelebA(fs_prefix + 'data', split='valid', download=False,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return ds_train, ds_val

    elif args.dataset == 'celeba32_sri':

        data_path = fs_prefix + 'data/{}/img_align_celeba'.format(args.dataset)
        cache_pkl = fs_prefix + 'data/{}/celeba_40000_32.pickle'.format(args.dataset)

        from data import SingleImagesFolderMTDataset
        import PIL
        import torchvision.transforms as transforms

        ds_train = SingleImagesFolderMTDataset(root=data_path,
                                            cache=cache_pkl,
                                            num_images=40000,
                                            transform=transforms.Compose([
                                                PIL.Image.fromarray,
                                                transforms.Resize(args.img_size),
                                                transforms.CenterCrop(args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        # TODO(nijkamp): create ds_val pickle
        ds_val = ds_train

        return ds_train, ds_val

    elif args.dataset == 'celeba64_sri':

        # wget https://www.dropbox.com/s/zjcpa1hrjxy9nne/celeba64_40000.pkl?dl=1

        data_path = fs_prefix + 'data/{}/img_align_celeba'.format(args.dataset)
        cache_pkl = fs_prefix + 'data/{}/celeba64_40000.pkl'.format(args.dataset)

        assert os.path.exists(cache_pkl)

        from data import SingleImagesFolderMTDataset
        import PIL
        import torchvision.transforms as transforms

        ds_train = SingleImagesFolderMTDataset(root=data_path,
                                            cache=cache_pkl,
                                            num_images=40000,
                                            transform=transforms.Compose([
                                                PIL.Image.fromarray,
                                                transforms.Resize(args.img_size),
                                                transforms.CenterCrop(args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        # TODO(nijkamp): create ds_val pickle
        ds_val = ds_train

        return ds_train, ds_val

    elif args.dataset == 'celeba64_sri_crop':

        # wget https://www.dropbox.com/s/9omncogiyaul54d/celeba_40000_64_center.pickle?dl=0

        data_path = fs_prefix + 'data/{}/img_align_celeba'.format(args.dataset)
        cache_pkl = fs_prefix + 'data/{}/celeba_40000_64_center.pickle'.format(args.dataset)

        from data import SingleImagesFolderMTDataset
        import PIL
        import torchvision.transforms as transforms

        ds_train = SingleImagesFolderMTDataset(root=data_path,
                                            cache=cache_pkl,
                                            num_images=40000,
                                            transform=transforms.Compose([
                                                PIL.Image.fromarray,
                                                transforms.Resize(args.img_size),
                                                transforms.CenterCrop(args.img_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        # TODO(nijkamp): create ds_val pickle
        ds_val = ds_train

        return ds_train, ds_val

    elif args.dataset == "cifar10":

        import torchvision.transforms as transforms
        ds_train = torchvision.datasets.CIFAR10(fs_prefix + 'data/{}'.format(args.dataset), train=True, download=True,
                                             transform=transforms.Compose([
                                                 transforms.Resize(args.img_size),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                             ]))
        ds_val = torchvision.datasets.CIFAR10(fs_prefix + 'data/{}'.format(args.dataset), train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(args.img_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
        return ds_train, ds_val

    else:
        raise ValueError(args.dataset)

##########################################################################################################

class Fid_calculator(object):

    def __init__(self, args, training_data):
        pfw.set_config(batch_size=args.batch_size, device=args.device)
        if training_data is None: 
            self.real_m, self.real_s = None, None
        else: 
            if os.path.exists("data/fid_stat_%s.npz" % args.dataset): 
                print("load precalculated FID distribution for training data.")
                data = np.load("data/fid_stat_%s.npz" % args.dataset)
                self.real_m, self.real_s = data['real_m'], data['real_s']
            else: 
                training_data = training_data.repeat(1,3 if training_data.shape[1] == 1 else 1,1,1)
                print("precalculate FID distribution for training data...")
                self.real_m, self.real_s = pfw.get_stats(training_data)
            print(self.real_m.mean(), self.real_s.mean())


    def fid(self, data): 
        if self.real_m is None: 
            return 0
        print(self.real_m.mean(), self.real_s.mean())
        data = data.repeat(1,3 if data.shape[1] == 1 else 1,1,1) 
        return pfw.fid(data, real_m=self.real_m, real_s=self.real_s)


def train(args, output_dir, path_check_point):

    #################################################
    ## preamble

    set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)
    tb_writer = SummaryWriter(log_dir=output_dir)
    job_id = int(args['job_id'])
    logger = setup_logging('job{}'.format(job_id), output_dir, console=True)
    logger.info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    #################################################
    ## data

    ds_train, ds_val = get_dataset(args)
    if args.incomplete_train is not None: 
        ds_train = IncompleteDataset(ds_train, args.incomplete_train, args.data_size)
    
    logger.info('len(ds_train)={}'.format(len(ds_train)))
    logger.info('len(ds_val)={}'.format(len(ds_val)))

    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=(args.incomplete_train is None), num_workers=0)
    # dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=0)

    if args.n_fid_samples > 0:
        args.n_fid_samples = min(len(ds_train), args.n_fid_samples)
        to_range_0_1 = lambda x: (x + 1.) / 2.
        # ds_fid = torch.stack([to_range_0_1(ds_train[i][0]) for i in range(len(ds_train))]).cpu()
        fid_calculator = Fid_calculator(args, 1)
    else: 
        fid_calculator = Fid_calculator(args, None)
    def plot(p, x):
        return torchvision.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=int(np.sqrt(args.batch_size)))

    #################################################
    ## model

    if args.dataset == "svhn":
        netG = _netG(args)
    elif args.dataset == "celeba" or args.dataset == "celeba_crop":
        netG = _netG_celeba(args)
    elif args.dataset == "mnist" or args.dataset == "mnist_ad":
        netG = _netG_mnist(args)
    elif args.dataset == "cifar10": 
        netG = _netG_cifar10(args)
    netF = _netF(args, nz=args.nz)

    netG.apply(weights_init_xavier)
    netF.apply(weights_init_xavier)

    netG = netG.to(device)
    netF = netF.to(device)

    logger.info(netG)
    logger.info(netF)

    def eval_flag():
        netG.eval()
        netF.eval()

    def train_flag():
        netG.train()
        netF.train()

    mse = nn.MSELoss(reduction='none')
    mse_mean = nn.MSELoss(reduction='mean')

    #################################################
    ## optimizer

    optG = torch.optim.Adam(netG.parameters(), lr=args.g_lr, weight_decay=args.g_decay, betas=(args.g_beta1, args.g_beta2))
    optF = torch.optim.Adam(netF.parameters(), lr=args.f_lr, weight_decay=args.f_decay, betas=(args.f_beta1, args.f_beta2))

    lr_scheduleG = torch.optim.lr_scheduler.ExponentialLR(optG, args.g_gamma)
    lr_scheduleF = torch.optim.lr_scheduler.ExponentialLR(optF, args.f_gamma)

    #################################################
    ## sampling

    def sample_p_0(n=args.batch_size, sig=1):
        return sig * torch.randn(*[n, args.nz, 1, 1]).to(device)


    def sample_langevin_post_z_with_flow(z, x, netG, netF, verbose=False, mask=None):
        z = z.clone().detach()
        z.requires_grad = True

        for i in range(args.g_l_steps):
            x_hat = netG(z)
            g_log_lkhd = mse(x_hat, x) #/ x.shape[0]
            if mask is not None: 
                g_log_lkhd *= mask 
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * g_log_lkhd.sum()
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            z1, logdet, _ = netF(torch.squeeze(z), objective=torch.zeros(int(z.shape[0])).to(device), init=False)
            prior_ll = -0.5 * (z1 ** 2)
            prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
            ll = prior_ll + logdet
            f_log_lkhd = -ll.sum()


            z_grad_f = torch.autograd.grad(f_log_lkhd, z)[0]
            z.data = z.data - 0.5 * args.g_l_step_size * args.g_l_step_size * (z_grad_g + z_grad_f)
            if args.g_l_with_noise:
                z.data += args.g_l_step_size * torch.randn_like(z).data

            z_grad_g_grad_norm = z_grad_g.view(z.shape[0], -1).norm(dim=1).mean()
            z_grad_f_grad_norm = z_grad_f.view(z.shape[0], -1).norm(dim=1).mean()

        if verbose:
            logger.info('Langevin posterior: MSE={:8.3f}, f_log_lkhd={:8.3f}, z_grad_g_grad_norm={:8.3f}, z_grad_f_grad_norm={:8.3f}'.format(g_log_lkhd.item(), f_log_lkhd.item(), z_grad_g_grad_norm, z_grad_f_grad_norm))


        return z.detach(), z_grad_g_grad_norm, z_grad_f_grad_norm

    #################################################
    ## train

    train_flag()


    # resume the training (1) for fine-tuning or (2) because of failure
    if path_check_point:
        ckp = torch.load(path_check_point)
        netG.load_state_dict(ckp['netG'])
        netF.load_state_dict(ckp['netF'])
        optG.load_state_dict(ckp['optG'])
        optF.load_state_dict(ckp['optF'])
        epoch_start=ckp['epoch']
        print("We resume the training from the last epoch.")
        fid_best = math.inf
    else:
        epoch_start=0
        print("This is a new training.")
        fid_best = math.inf

    fid = 0.0
    z_fixed = sample_p_0()
    x_fixed = next(iter(dataloader_train))[0].to(device)

    stats = {
        'loss_g':[],
        'loss_e':[],
        'en_neg':[],
        'en_pos':[],
        'grad_norm_g':[],
        'grad_norm_e':[],
        'z_e_grad_norm':[],
        'z_g_grad_norm':[],
        'z_e_k_grad_norm':[],
        'fid':[],
    }
    interval = []

    for epoch in range(epoch_start, args.n_epochs):

        if epoch_start == 0 or epoch > epoch_start:
            for i, x_input in enumerate(dataloader_train, 0):

                if args.incomplete_train is not None: 
                    x_gt, y, masks = x_input
                    x = x_gt * masks 
                else: 
                    x, y = x_input
                    masks = torch.ones(1)
                train_flag()

                x = x.to(device)
                masks = masks.to(device)
                batch_size = x.shape[0]

                # Initialize chain
                z_g_0 = sample_p_0(n=batch_size)
                z_f_0 = sample_p_0(n=batch_size)

                z_g_k, z_g_grad_norm, z_f_grad_norm = sample_langevin_post_z_with_flow(Variable(z_g_0), x, netG, netF, verbose=False, mask=masks)

                # Learn generator
                optG.zero_grad()

                x_hat = netG(z_g_k.detach())
                loss_g = (mse(x_hat, x) * masks).sum() / batch_size
                loss_g.backward()
                grad_norm_g = get_grad_norm(netG.parameters())
                if args.g_is_grad_clamp:
                    torch.nn.utils.clip_grad_norm(netG.parameters(), opt.g_max_norm)
                optG.step()
                #if jj % 20 == 0:
                #    logger.info('Train generator: loss_g={:8.3f}'.format(loss_g))


                # Learn prior flow
                optF.zero_grad()

                z1, logdet, _ = netF(torch.squeeze(z_g_k), objective=torch.zeros(int(z_g_k.shape[0])).to(device), init=False)
                prior_ll = -0.5 * (z1 ** 2)
                prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
                ll = prior_ll + logdet
                loss_f = -ll.mean()
                loss_f.backward()

                if args.f_is_grad_clamp:
                    torch.nn.utils.clip_grad_norm_(netF.parameters(), args.f_max_norm)
                optF.step()


                # Printout
                if i % args.n_printout == 0:
                        # x_0 = netG(z_e_0)
                        # x_k = netG(z_e_k)

                        # en_neg_2 = energy(netE(z_e_k)).mean()
                        # en_pos_2 = energy(netE(z_g_k)).mean()

                        # prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_e_k.mean(), z_e_k.std(), z_e_k.abs().max())
                        # posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_g_k.mean(), z_g_k.std(), z_g_k.abs().max())

                        logger.info('{} {:5d}/{:5d} {:5d}/{:5d} '.format(job_id, epoch, args.n_epochs, i, len(dataloader_train)) +
                            'loss_g={:8.3f}, '.format(loss_g) +
                            'loss_f={:8.3f}, '.format(loss_f) +
                            #'|grad_g|={:8.2f}, '.format(grad_norm_g) +
                            '|z_g_grad|={:7.3f}, '.format(z_g_grad_norm) +
                            '|z_f_grad|={:7.3f}, '.format(z_f_grad_norm) +
                            #'posterior_moments={}, '.format(posterior_moments) +
                            'fid={:8.2f}, '.format(fid) +
                            'fid_best={:8.2f}'.format(fid_best))

                        num_step = epoch*len(dataloader_train)+i
                        tb_writer.add_scalar("train/loss_g", loss_g, num_step)
                        tb_writer.add_scalar("train/loss_f", loss_f, num_step)
                        tb_writer.add_scalar("train/z_g_grad_norm", z_g_grad_norm, num_step)
                        tb_writer.add_scalar("train/z_f_grad_norm", z_f_grad_norm, num_step)
                        tb_writer.add_scalar("train/fid_best", fid_best, num_step)

            # Schedule
            # lr_scheduleE.step(epoch=epoch)
            lr_scheduleG.step(epoch=epoch)
            lr_scheduleF.step(epoch=epoch)

        # Metrics
        if epoch == args.n_epochs or epoch % args.n_metrics == 0:

            # generate
            if args.n_fid_samples > 0:
                try:
                    def sample_x():
                        z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
                        z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(int(z_sample.shape[0])).to(device), reverse=True, return_obj=False)
                        x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
                        x_samples[x_samples == float("inf")] = 1
                        x_samples[x_samples == -float("inf")] = 0
                        x_samples[x_samples != x_samples] = 0
                        x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()
                        return x_samples
                    with torch.no_grad():
                        x_samples = torch.cat([sample_x() for _ in range(int(args.n_fid_samples / args.batch_size))]).detach()
                    fid = fid_calculator.fid(x_samples)

                except Exception as e:
                    print(e)
                    logger.critical(e, exc_info=True)
                    logger.info('FID failed')
                    fid = 10000
                if fid < fid_best:
                    fid_best = fid
                logger.info('fid={}'.format(fid))
                tb_writer.add_scalar("train/fid", fid, epoch)
            if args.incomplete_train is not None: 
                total_loss, recovery_loss, unmasked_loss = 0, 0, 0
                for i, x_input in tqdm(enumerate(dataloader_train, 0), "reconstruct", len(dataloader_train), False):
                    x_gt, y, masks = x_input
                    x_gt = x_gt.to(device)
                    masks = masks.to(device)
                    x = x_gt * masks 
                    batch_size = x.shape[0]

                    # Initialize chain
                    z_g_0 = sample_p_0(n=batch_size)
                    z_g_k, z_g_grad_norm, z_f_grad_norm = sample_langevin_post_z_with_flow(Variable(z_g_0), x, netG, netF, verbose=False, mask=masks)
                    x_hat = netG(z_g_k.detach())
                    loss = mse(x_hat, x_gt).detach()
                    num_unmasked_pixel = masks.sum()
                    total_loss += loss.mean() * batch_size
                    recovery_loss += (loss * (~masks)).sum() / (batch_size * 3 * 64 * 64 - num_unmasked_pixel) * batch_size
                    unmasked_loss += (loss * masks).sum() / num_unmasked_pixel * batch_size
                total_loss /= len(ds_train)
                recovery_loss /= len(ds_train)
                unmasked_loss /= len(ds_train)
                logger.info("Recovery check: total/recovery/unmasked loss: %.4f/%.4f/%.4f" % (total_loss*10000, recovery_loss*10000, unmasked_loss*10000))
                tb_writer.add_scalar("incomplete/total_loss", total_loss, epoch)
                tb_writer.add_scalar("incomplete/recovery_loss", recovery_loss, epoch)
                tb_writer.add_scalar("incomplete/unmasked_loss", unmasked_loss, epoch)

            # plot
            with torch.no_grad():

                # x_0 = netG(z_f_0)

                # z_sample = []
                # z_shapes = []
                # for ib in range(args.f_n_block):
                #     z_shapes.append((args.f_in_channel, 1, 1))
                # for z in z_shapes:
                #     z_new = torch.randn(args.batch_size, *z) * args.temp
                #     z_sample.append(z_new.to(device))
                # z_f_k = netF.reverse(z_sample)

                z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
                z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(int(z_sample.shape[0])).to(device), reverse=True, return_obj=False)
                x_k = netG( torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
                plot('{}/samples/{:>06d}_x_z_flow_prior.png'.format(output_dir, epoch), x_k)

                if args.incomplete_train is not None: 
                    torchvision.utils.save_image(torch.clamp(x_hat, -1., 1.), '{}/samples/reconstruct_{:>06d}.png'.format(output_dir, epoch), normalize=True, nrow=int(np.sqrt(args.batch_size)))
                    torchvision.utils.save_image(torch.clamp(x, -1., 1.), '{}/samples/gt_mask_{:>06d}.png'.format(output_dir, epoch), normalize=True, nrow=int(np.sqrt(args.batch_size)))
                    x_combined = torch.where(masks, x, torch.clamp(x_hat, -1., 1.))
                    torchvision.utils.save_image(x_combined, '{}/samples/combined_{:>06d}.png'.format(output_dir, epoch), normalize=True, nrow=int(np.sqrt(args.batch_size)))

        # Plot
        # if epoch % args.n_plot == 0:
        #
        #     batch_size_fixed = x_fixed.shape[0]
        #
        #     z_g_0 = sample_p_0(n=batch_size_fixed)
        #     z_f_0 = sample_p_0(n=batch_size_fixed)
        #
        #     z_g_k, z_g_grad_norm, z_f_grad_norm = sample_langevin_post_z_with_flow(Variable(z_g_0), x_fixed, netG, netF)
        #
        #
        #     # z_e_k, z_e_k_grad_norm = sample_langevin_prior_z(Variable(z_e_0), netE)
        #
        #     # x_0 = netG(z_f_0)
        #     # z_f_k = netF.reverse(z_f_0)
        #     # x_k = netG(z_f_k)
        #
        #     z_sample = []
        #     z_shapes = []
        #     for ib in range(args.f_n_block):
        #         z_shapes.append((args.f_in_channel, 1, 1))
        #     for z in z_shapes:
        #         z_new = torch.randn(args.batch_size, *z) #* args.temp
        #         z_sample.append(z_new.to(device))
        #
        #     z_f_k = netF.reverse(z_sample)
        #
        #
        #
        #
        #
        #     with torch.no_grad():
        #         # plot('{}/samples/{:>06d}_{:>06d}_x_gt_fixed.png'.format(output_dir, epoch, i), x_fixed)
        #         # plot('{}/samples/{:>06d}_{:>06d}_x_z_pos.png'.format(output_dir, epoch, i), netG(z_g_k))
        #         # plot('{}/samples/{:>06d}_{:>06d}_x_z_noise_prior.png'.format(output_dir, epoch, i), netG(z_f_0))
        #         plot('{}/samples/{:>06d}_{:>06d}_x_z_flow_prior.png'.format(output_dir, epoch, i), netG(z_f_k))
        #         # plot('{}/samples/{:>06d}_{:>06d}_x_z_fixed_noise_prior.png'.format(output_dir, epoch, i), netG(z_fixed))

        # Ckpt
        if epoch == args.n_epochs or epoch % args.n_ckpt == 0:

            save_dict = {
                'epoch': epoch,
                'netF': netF.state_dict(),
                'optF': optF.state_dict(),
                'netG': netG.state_dict(),
                'optG': optG.state_dict(),
            }
            torch.save(save_dict, '{}/ckpt/ckpt_{:>06d}.pth'.format(output_dir, epoch))

        # Early exit
        # if epoch > 10 and loss_g > 300:
        #     logger.info('early exit condition 1: epoch > 10 and loss_g > 300')
        #     return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
        #     return

        # if epoch > 40 and fid > 100:
        #     logger.info('early exit condition 2: epoch > 40 and fid > 100')
        #     return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
        #     return

    #return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
    logger.info('done')


##########################################################################################################
## Metrics

def is_xsede():
    import socket
    return 'psc' in socket.gethostname()

#################################################
## test

def print_all_latent(netG, netF, args): 
    to_range_0_1 = lambda x: (x + 1.) / 2.
    z_sample = torch.randn(20, args.nz, 1, 1).to(args.devices)
    # z_sample = torch.randn(1, args.nz, 1, 1).to(args.devices)
    # z_sample2 = torch.randn(1, args.nz, 1, 1).to(args.devices)
    # z_sample = torch.randn(1, args.nz, 1, 1).to(args.devices)
    # z_sample2 = torch.randn(1, args.nz, 1, 1).to(args.devices)
    # z_line = []
    # for p in np.linspace(0, 1, 10):
    #     z_line.append(z_sample * p + z_sample2 * (1-p))
    # z_sample = torch.cat(z_line)
    z_f_k, z_f_all = netF(torch.squeeze(z_sample), objective=torch.zeros(
        int(z_sample.shape[0])).to(args.devices), reverse=True, return_all=True)
    z_f_k = torch.cat([z_sample.squeeze()] + z_f_all)
    x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
    x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()
    path = 'output/generate_x_flow_rand_celeba.png'.format(args.output_dir)
    torchvision.utils.save_image(x_samples, path, normalize=True, nrow=20)

def print_grid(netG, netF, args): 
    to_range_0_1 = lambda x: (x + 1.) / 2.
    z_line = []
    for x, y in itertools.product(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20)):
        z_line.append([x,y])
    z_sample = torch.from_numpy(np.array(z_line)).float().to(args.devices)
    z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(
        int(z_sample.shape[0])).to(args.devices), reverse=True)
    x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
    x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()
    path = 'output/generate_grid.png'.format(args.output_dir)
    torchvision.utils.save_image(x_samples, path, normalize=True, nrow=20)

def print_intepolation(netG, netF, args): 
    to_range_0_1 = lambda x: (x + 1.) / 2.
    z_sample = torch.randn(20, args.nz, 1, 1).to(args.devices)
    z_sample2 = torch.randn(20, args.nz, 1, 1).to(args.devices)
    z_line = []
    for p in np.linspace(0, 1, 10):
        z_line.append(z_sample * p + z_sample2 * (1-p))
    z_sample = torch.cat(z_line)
    z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(
        int(z_sample.shape[0])).to(args.devices), reverse=True)
    x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
    x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()
    path = 'output/generate_intepolation.png'.format(args.output_dir)
    torchvision.utils.save_image(x_samples, path, normalize=True, nrow=20)

# AUC anormaly detection added by Jerry
def test(args, output_dir, path_check_point):

    #################################################
    ## preamble

    if args.device > 0:
        set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)
    args.devices = device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    #################################################
    ## model

    if args.dataset == "svhn":
        netG = _netG(args)
    elif args.dataset == "celeba" or args.dataset == "celeba_crop":
        netG = _netG_celeba(args)
    elif args.dataset == "mnist" or args.dataset == "mnist_ad":
        netG = _netG_mnist(args)
    netF = _netF(args, nz=args.nz)

    ckp = torch.load(path_check_point)
    netG.load_state_dict(ckp['netG'])
    netF.load_state_dict(ckp['netF'])

    netG = netG.to(device)
    netF = netF.to(device)

    def eval_flag():
        netG.eval()
        netF.eval()

    #################################################
    ## test

    to_range_0_1 = lambda x: (x + 1.) / 2.
    ds_train, ds_test = get_dataset(args)
    real_m = None
    args.tasks = ["incomplete"]
    if "generate_all" in args.tasks: 
        print_all_latent(netG, netF, args)
        print_grid(netG, netF, args)
        n = args.n_fid_samples
        print('computing fid with {} samples'.format(n))
        # eval_flag()
        to_range_0_1 = lambda x: (x + 1.) / 2.
        def sample_x():
            z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
            z_f_k, z_f_all = netF(torch.squeeze(z_sample), objective=torch.zeros(
            int(z_sample.shape[0])).to(device), reverse=True, return_obj=False, return_all=True)
            z_f_all = [z_sample.squeeze()] + z_f_all
            res = []
            for z in z_f_all: 
                x_samples = netG(torch.reshape(z, (z.shape[0], z.shape[1], 1, 1)))
                res.append(to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu())
            return res

        pfw.set_config(batch_size=args.batch_size, device=args.device)
        ds_fid = torch.stack([to_range_0_1(ds_train[i][0]).cpu() for i in range(len(ds_train))])
        ds_fid = ds_fid.repeat(1,3 if ds_fid.shape[1] == 1 else 1,1,1)
        real_m, real_s = pfw.get_stats(ds_fid)
        res = [sample_x() for _ in range(int(n / args.batch_size))]
        for i, x in enumerate(res[0]):
            x_sample = torch.cat([r[i] for r in res])
            x_sample = x_sample.repeat(1,3 if x_sample.shape[1] == 1 else 1,1,1) 
            fid = pfw.fid(x_sample, real_m=real_m, real_s=real_s)
            print('layer %d: fid=%.4f'%(i, fid))

    if "generate" in args.tasks: 
        n = args.n_fid_samples
        print('computing fid with {} samples'.format(n))
        # eval_flag()
        def sample_x(return_all=False):
            z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
            z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(
            int(z_sample.shape[0])).to(device), reverse=True, return_obj=False)
            x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
            res =  to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()
            return res

        if real_m is None: 
            pfw.set_config(batch_size=args.batch_size, device=args.device)
            ds_fid = torch.stack([to_range_0_1(ds_train[i][0]).cpu() for i in range(len(ds_train))])
            ds_fid = ds_fid.repeat(1,3 if ds_fid.shape[1] == 1 else 1,1,1)
            real_m, real_s = pfw.get_stats(ds_fid)
        x_samples = torch.cat([sample_x() for _ in range(int(n / args.batch_size))])
        x_samples = x_samples.repeat(1,3 if x_samples.shape[1] == 1 else 1,1,1)
        fid = pfw.fid(x_samples, real_m=real_m, real_s=real_s)
        print('fid=%.4f'%(fid))
        
    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    mse = nn.MSELoss(reduction='none')
    def sample_langevin_post_z_with_flow(z, x, netG, netF, mask=None):
        z = z.clone().detach()
        z.requires_grad = True

        g_l_steps_testing = args.g_l_steps * 20
        g_l_step_size_testing = args.g_l_step_size

        for i in range(g_l_steps_testing):
            x_hat = netG(z)
            g_log_lkhd = mse(x_hat, x) #/ x.shape[0]
            if mask is not None: 
                g_log_lkhd *= mask 
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * g_log_lkhd.sum()
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            z1, logdet, _ = netF(torch.squeeze(z), objective=torch.zeros(int(z.shape[0])).to(device), init=False)
            prior_ll = -0.5 * (z1 ** 2)
            prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
            ll = prior_ll + logdet
            f_log_lkhd = -ll.sum()
            z_grad_f = torch.autograd.grad(f_log_lkhd, z)[0]

            z.data = z.data - 0.5 * g_l_step_size_testing * g_l_step_size_testing * (z_grad_g + z_grad_f)
            #if args.g_l_with_noise:
            #    z.data += args.g_l_step_size * torch.randn_like(z).data
            z_grad_g_grad_norm = z_grad_g.view(x.shape[0], -1).norm(dim=1).mean()
            z_grad_f_grad_norm = z_grad_f.view(x.shape[0], -1).norm(dim=1).mean()

        return z.detach(), z_grad_g_grad_norm, z_grad_f_grad_norm

    if "incomplete" in args.tasks: 

        print("do incomplete")
        # generate_size = args.batch_size // 2
        # masks = np.ones((generate_size * 2, args.img_size, args.img_size))
        # patch_size, num_patch = 8, 20 
        # rad1 = torch.randint(0, args.img_size-patch_size, size=(generate_size, num_patch, 2))
        # for i in range(generate_size): 
        #     for j in range(num_patch): 
        #         masks[i, rad1[i, j, 0]:rad1[i, j, 0]+patch_size, rad1[i, j, 1]:rad1[i, j, 1]+patch_size] = 0

        # # random large square: 
        # patch_size, num_patch = 32, 1 
        # rad2 = torch.randint(0, args.img_size-patch_size, size=(generate_size, num_patch, 2))
        # for i in range(generate_size): 
        #     for j in range(num_patch): 
        #         masks[i+generate_size, rad2[i, j, 0]:rad2[i, j, 0]+patch_size, rad2[i, j, 1]:rad2[i, j, 1]+patch_size] = 0
        # masks = torch.tensor(masks).to(device).unsqueeze(1).repeat(1,3,1,1)

        # constant large square: 
        masks = np.ones((args.batch_size, 3, args.img_size, args.img_size))
        masks[:, :, 20:60, 12:52] = 0
        masks = torch.tensor(masks).to(device)

        rec_errors,x_samples = [], []
        for i, (x, y) in tqdm(enumerate(dataloader_test, 0), leave=False):
            x = x.to(device)
            np.save("output/incomplete_truth.npy", x.cpu().data.numpy())
            print(x.shape)
            break 
        for i in range(5):
            z_g_0 = torch.randn(x.shape[0], args.nz, 1, 1).to(device)
            z_g_k = sample_langevin_post_z_with_flow(z_g_0, x, netG, netF, mask=None if i==0 else masks)[0]
            x_hat = netG(z_g_k.detach())
            rec_errors.append(mse(x_hat, x).mean())
            print(rec_errors[-1])
            x_samples.append(x_hat.clamp(min=-1., max=1.).detach().cpu())
        path = 'output/incomplete.png'.format(args.output_dir)
        x, masks = x.cpu(), masks.cpu()
        torchvision.utils.save_image(torch.cat([x, x * masks] + x_samples), path, normalize=True, nrow=args.batch_size)

        for i in range(1, 10): 
            x_frac = x.clone()
            x_frac[:, :, 20:60, 12:52] = x_samples[i][:, :, 20:60, 12:52]
            x_samples[i] = x_frac
        path = 'output/incomplete_true.png'.format(args.output_dir)
        torchvision.utils.save_image(torch.cat([x, x * masks] + x_samples), path, normalize=True, nrow=args.batch_size)
        
        

    from sklearn.metrics import precision_recall_curve, auc
    import matplotlib.pyplot as plt 

    Y_hat, Y, rec_errors, Z, Zg = [], [], [], [], []
    # print('anomaly detection starts for %i' % args.abnormal)
    for i, (x, y) in tqdm(enumerate(dataloader_test, 0), leave=False):
        x = x.to(device)
        z_g_0 = torch.randn(x.shape[0], args.nz, 1, 1).to(device)
        z_g_k = sample_langevin_post_z_with_flow(z_g_0, x, netG, netF)[0]
        x_hat = netG(z_g_k.detach())
        z1, logdet, _ = netF(torch.squeeze(z_g_k), objective=torch.zeros(int(z_g_k.shape[0])).to(device), init=False)
        # x_hat = to_range_0_1(x_hat).clamp(min=0., max=1.)
        rec_errors.append(mse(x_hat, x).sum())
        Y.append(y) 
        if "visual_2dim" in args.tasks:
            Z.append(z1)
            Zg.append(z_g_k)
        if args.abnormal is not None: 
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * torch.sum(torch.pow(x - x_hat, 2), (3,2,1))
            f_log_lkhd = -((-0.5 * (z1 ** 2)).flatten(1).sum(-1) + np.log(2 * np.pi) + logdet)
            Y_hat.append(g_log_lkhd + f_log_lkhd)
        if i == 2: 
            torchvision.utils.save_image(torch.cat([x[:20], x_hat[:20]]), "output/2dim_recon.png", normalize=True, nrow=20)
            break

    if "visual_2dim" in args.tasks:

        loc = np.concatenate([z.cpu().data.numpy() for z in Z])
        print(loc.shape)
        label = np.concatenate([y.cpu().data.numpy() for y in Y])
        colormap = ["r", "g", "b", "c", "m", "y", "lime", "gold", "pink", "tomato"]
        color = [colormap[i] for i in label]
        plt.scatter(loc[:,0], loc[:,1], c=color, s=1)
        plt.savefig("output/2dim_vis_z1.png")

        plt.clf()
        loc = np.concatenate([z.cpu().data.numpy() for z in Zg])
        plt.scatter(loc[:,0,0,0], loc[:,1,0,0], c=color, s=1)
        plt.savefig("output/2dim_vis_zg.png")

    if args.abnormal != -1: 
        Y_raw = Y
        Y = np.concatenate([y.cpu().data.numpy() for y in Y]) == args.abnormal 
        # print("[%d / %d] abnormal used." % (Y.sum(), len(Y)))
        Y_hat = np.concatenate([y.cpu().data.numpy() for y in Y_hat])
        precision, recall, thresholds = precision_recall_curve(Y, Y_hat)
        auc_ = auc(recall, precision)
        print("AUC = ", auc_, np.sum(Y))
        plt.plot(recall, precision)
        plt.savefig("output/abnormal_%s_auc.png" % args.abnormal)
    
    recon_error = float(torch.sum(torch.stack(rec_errors)).cpu().data.numpy()) / len(ds_test) / 3 / args.img_size / args.img_size
    print('reconstruction error={}'.format(recon_error))

##########################################################################################################
## Other

def get_grad_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def print_gpus():
    os.system('nvidia-smi -q -d Memory > tmp')
    tmp = open('tmp', 'r').readlines()
    for l in tmp:
        print(l, end = '')

def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d

##########################################################################################################
## Main

def main():

    opt = {'job_id': int(0), 'status': 'open'}
    args = parse_args()
    args = pygrid.overwrite_opt(args, opt)
    args = to_named_dict(args)
    path_check_point = None if args.load_checkpoint == "" else args.load_checkpoint
    output_dir = pygrid.get_output_dir(get_exp_id(__file__), fs_prefix='./') if args.output_dir == "default" else ("output/train_%s/" % args.dataset + args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/samples')
        os.makedirs(output_dir + '/ckpt')

    if args.mode == "train":
        # training mode
        copy_source(__file__, output_dir)
        train(args, output_dir, path_check_point)
    elif args.mode == "test":
        # testing mode
        test(args, output_dir, path_check_point)

if __name__ == '__main__':
    main()