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

import torch
from torch import cuda
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

import torchvision
import torchvision.transforms as transforms

import pygrid

from model import _netG, _netE, _netF, weights_init_xavier

##########################################################################################################
## Parameters

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')

    parser.add_argument('--dataset', type=str, default='svhn', choices=['svhn', 'celeba', 'celeba_crop', 'celeba32_sri', 'celeba64_sri', 'celeba64_sri_crop'])
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--batch_size', default=100, type=int)

    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', default=3)

    #parser.add_argument('--nez', default=1, help='size of the output of ebm')
    parser.add_argument('--ngf', default=64, help='feature dimensions of generator')
    #parser.add_argument('--ndf', default=200, help='feature dimensions of ebm')

    # parser.add_argument('--e_prior_sig', type=float, default=1, help='prior of ebm z')
    # #parser.add_argument('--e_init_sig', type=float, default=1, help='sigma of initial distribution')
    # parser.add_argument('--e_activation', type=str, default='gelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    # parser.add_argument('--e_activation_leak', type=float, default=0.2)
    # parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    # parser.add_argument('--e_l_steps', type=int, default=60, help='number of langevin steps')
    # parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    # parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    # parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')

    parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='prior of factor analysis')
    parser.add_argument('--g_activation', type=str, default='lrelu')
    parser.add_argument('--g_activation_leak', type=float, default=0.2)
    parser.add_argument('--g_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of langevin') # 0.1
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--g_batchnorm', default=False, type=bool, help='batch norm')

    # parser.add_argument('--f_in_channel', default=100, help='in channel of flow model')
    # parser.add_argument('--f_n_flow', default=30, type=int, help='number of flows in each block')
    # parser.add_argument('--f_n_block', default=1, type=int, help='number of blocks')
    # parser.add_argument('--f_affine', default=True, type=bool, help='use affine coupling instead of additive')
    # parser.add_argument('--f_conv_lu', default=True, type=bool, help='use LU decomposed version instead of plain convolution')
    # parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
    # parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')

    parser.add_argument('--n_levels', default=1, type=int, help='')
    parser.add_argument('--depth', default=5, type=int, help='') # 10
    parser.add_argument('--flow_permutation', default=2, type=int, help='')
    parser.add_argument('--width', default=64, type=int, help='')
    parser.add_argument('--flow_coupling', default=1, type=int, help='')

    # parser.add_argument('--e_lr', default=0.00002, type=float)
    parser.add_argument('--g_lr', default=0.0004, type=float)
    parser.add_argument('--f_lr', default=0.0004, type=float) # 0.0002

    # parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--f_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')

    #parser.add_argument('--e_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--f_max_norm', type=float, default=100, help='max norm allowed')

    #parser.add_argument('--e_decay', default=0, help='weight decay for ebm')
    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')
    parser.add_argument('--f_decay', default=0, help='weight decay for flow')

    #parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--g_gamma', default=0.998, help='lr decay for gen')
    parser.add_argument('--f_gamma', default=0.998, help='lr decay for flow')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    #parser.add_argument('--e_beta1', default=0.5, type=float)
    #parser.add_argument('--e_beta2', default=0.999, type=float)

    parser.add_argument('--f_beta1', default=0.5, type=float)
    parser.add_argument('--f_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for') # TODO(nijkamp): set to >100
    # parser.add_argument('--n_epochs', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=20, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')

    parser.add_argument('--n_ckpt', type=int, default=1, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')
    #
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')

    parser.add_argument('--n_fid_samples', type=int, default=50000) # TODO(nijkamp): we used 40,000 in short-run inference
    # parser.add_argument('--n_fid_samples', type=int, default=1000)

    return parser.parse_args()


def create_args_grid():
    # TODO add your enumeration of parameters here

    e_lr = [0.00002]
    e_l_step_size = [0.4]
    e_init_sig = [1.0]
    e_l_steps = [30,50,60]
    e_activation = ['lrelu']

    g_llhd_sigma = [0.3]
    g_lr = [0.0001]
    g_l_steps = [20]
    g_activation = ['lrelu']

    ngf = [64]
    ndf = [200]

    args_list = [e_lr, e_l_step_size, e_init_sig, e_l_steps, e_activation, g_llhd_sigma, g_lr, g_l_steps, g_activation, ngf, ndf]

    opt_list = []
    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        opt_args = {
            'e_lr': args[0],
            'e_l_step_size': args[1],
            'e_init_sig': args[2],
            'e_l_steps': args[3],
            'e_activation': args[4],
            'g_llhd_sigma': args[5],
            'g_lr': args[6],
            'g_l_steps': args[7],
            'g_activation': args[8],
            'ngf': args[9],
            'ndf': args[10],
        }
        # TODO add your result metric here
        opt_result = {'fid_best': 0.0, 'fid': 0.0, 'mse': 0.0}

        opt_list += [merge_dicts(opt_job, opt_args, opt_result)]

    return opt_list


def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    job_opt['fid_best'] = job_stats['fid_best']
    job_opt['fid'] = job_stats['fid']
    job_opt['mse'] = job_stats['mse']


##########################################################################################################
## Data

def get_dataset(args):

    fs_prefix = './' if not is_xsede() else '/pylon5/ac561ep/enijkamp/ebm_prior/'

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

        ds_train = torchvision.datasets.CelebA(fs_prefix + 'data/{}/train'.format(args.dataset), split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(args.img_size),
                                                        transforms.CenterCrop(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        ds_val = torchvision.datasets.CelebA(fs_prefix + 'data/{}/val'.format(args.dataset), split='valid', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Resize(args.img_size),
                                                        transforms.CenterCrop(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return ds_train, ds_val

    if args.dataset == 'celeba_crop':

        crop = lambda x: transforms.functional.crop(x, 45, 25, 173-45, 153-25)

        import torchvision.transforms as transforms

        ds_train = torchvision.datasets.CelebA(fs_prefix + 'data/{}/train'.format(args.dataset), split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        ds_val = torchvision.datasets.CelebA(fs_prefix + 'data/{}/val'.format(args.dataset), split='valid', download=True,
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

    else:
        raise ValueError(args.dataset)

##########################################################################################################

def train(args_job, output_dir_job, output_dir, return_dict):

    #################################################
    ## preamble

    args = parse_args()
    args = pygrid.overwrite_opt(args, args_job)
    args = to_named_dict(args)

    set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)

    makedirs_exp(output_dir)

    job_id = int(args['job_id'])

    logger = setup_logging('job{}'.format(job_id), output_dir, console=True)
    logger.info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    #################################################
    ## data

    ds_train, ds_val = get_dataset(args)
    logger.info('len(ds_train)={}'.format(len(ds_train)))
    logger.info('len(ds_val)={}'.format(len(ds_val)))

    dataloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # dataloader_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=True, num_workers=0)

    assert len(ds_train) >= args.n_fid_samples
    to_range_0_1 = lambda x: (x + 1.) / 2.
    ds_fid = np.array(torch.stack([to_range_0_1(ds_train[i][0]) for i in range(len(ds_train))]).cpu().numpy())
    logger.info('ds_fid.shape={}'.format(ds_fid.shape))

    def plot(p, x):
        return torchvision.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=int(np.sqrt(args.batch_size)))

    #################################################
    ## model

    netG = _netG(args)
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

    mse = nn.MSELoss(reduction='sum')

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


    def sample_langevin_post_z_with_flow(z, x, netG, netF, verbose=False):
        z = z.clone().detach()
        z.requires_grad = True

        for i in range(args.g_l_steps):
            x_hat = netG(z)
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat, x) #/ x.shape[0]
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            z1, logdet, _ = netF(torch.squeeze(z), objective=torch.zeros(int(z.shape[0])).to(device), init=False)
            prior_ll = -0.5 * (z1 ** 2)
            prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
            ll = prior_ll + logdet
            #f_log_lkhd = -ll.mean()
            f_log_lkhd = -ll.sum()



            #log_p, logdet, _ = netF(z)
            #logdet = logdet.mean()
            #f_log_lkhd, _, _ = calc_loss(log_p, logdet, args.f_in_channel, n_bins=2.0 ** args.n_bits)

            z_grad_f = torch.autograd.grad(f_log_lkhd, z)[0]

            z.data = z.data - 0.5 * args.g_l_step_size * args.g_l_step_size * (z_grad_g + z_grad_f)
            if args.g_l_with_noise:
                z.data += args.g_l_step_size * torch.randn_like(z).data

            z_grad_g_grad_norm = z_grad_g.view(args.batch_size, -1).norm(dim=1).mean()
            z_grad_f_grad_norm = z_grad_f.view(args.batch_size, -1).norm(dim=1).mean()

        if verbose:
            logger.info('Langevin posterior: MSE={:8.3f}, f_log_lkhd={:8.3f}, z_grad_g_grad_norm={:8.3f}, z_grad_f_grad_norm={:8.3f}'.format(g_log_lkhd.item(), f_log_lkhd.item(), z_grad_g_grad_norm, z_grad_f_grad_norm))


        return z.detach(), z_grad_g_grad_norm, z_grad_f_grad_norm

    #################################################
    ## fid

    def get_fid(n):

        assert n <= ds_fid.shape[0]

        logger.info('computing fid with {} samples'.format(n))

        try:
            eval_flag()

            def sample_x():

                z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
                z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(int(z_sample.shape[0])).to(device),
                             reverse=True, return_obj=False)

                x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
                x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()

                return x_samples

            x_samples = torch.cat([sample_x() for _ in range(int(n / args.batch_size))]).numpy()
            fid = compute_fid_nchw(args, ds_fid, x_samples)
            return fid

        except Exception as e:
            print(e)
            logger.critical(e, exc_info=True)
            logger.info('FID failed')

        finally:
            train_flag()




    #################################################
    ## train

    train_flag()

    fid = 0.0
    fid_best = math.inf

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

    for epoch in range(args.n_epochs):

        for i, (x, y) in enumerate(dataloader_train, 0):

            train_flag()

            x = x.to(device)
            batch_size = x.shape[0]

            # Initialize chain
            z_g_0 = sample_p_0(n=batch_size)
            z_f_0 = sample_p_0(n=batch_size)


            z_g_k, z_g_grad_norm, z_f_grad_norm = sample_langevin_post_z_with_flow(Variable(z_g_0), x, netG, netF, verbose=False)


            # Learn generator

            optG.zero_grad()

            x_hat = netG(z_g_k.detach())
            loss_g = mse(x_hat, x) / batch_size
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


            # log_p, logdet, _ = netF(z_g_k.detach() + torch.rand_like(z_g_k) / 2.0 ** args.n_bits)
            # logdet = logdet.mean()
            # loss_f, _, _ = calc_loss(log_p, logdet, args.f_in_channel, n_bins=2.0 ** args.n_bits)
            # loss_f = loss_f / batch_size
            # loss_f.backward()
            # #grad_norm_f = get_grad_norm(netF.parameters())

            if args.f_is_grad_clamp:
                 torch.nn.utils.clip_grad_norm_(netF.parameters(), args.f_max_norm)
            optF.step()
            #if jj % 20 == 0:
            #    logger.info('Train flow: loss_f={:8.3f}'.format(loss_f))


            # Printout
            if i % args.n_printout == 0:
                with torch.no_grad():

                    x_0 = netG(z_f_0)

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
                    #

                    x_k = netG( torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))

                    plot('{}/samples/{:>06d}_x_z_flow_prior.png'.format(output_dir, epoch), x_k)
                    #torchvision.utils.save_image(x_k, '{}/epoch_{}.png'.format(output_dir, epoch), nrow=int(10))

                    # x_0 = netG(z_e_0)
                    # x_k = netG(z_e_k)

                    # en_neg_2 = energy(netE(z_e_k)).mean()
                    # en_pos_2 = energy(netE(z_g_k)).mean()

                    # prior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_e_k.mean(), z_e_k.std(), z_e_k.abs().max())
                    posterior_moments = '[{:8.2f}, {:8.2f}, {:8.2f}]'.format(z_g_k.mean(), z_g_k.std(), z_g_k.abs().max())

                    logger.info('{} {:5d}/{:5d} {:5d}/{:5d} '.format(job_id, epoch, args.n_epochs, i, len(dataloader_train)) +
                        'loss_g={:8.3f}, '.format(loss_g) +
                        'loss_f={:8.3f}, '.format(loss_f) +
                        #'|grad_g|={:8.2f}, '.format(grad_norm_g) +
                        '|z_g_grad|={:7.3f}, '.format(z_g_grad_norm) +
                        '|z_f_grad|={:7.3f}, '.format(z_f_grad_norm) +
                        #'posterior_moments={}, '.format(posterior_moments) +
                        'fid={:8.2f}, '.format(fid) +
                        'fid_best={:8.2f}'.format(fid_best))

        # Schedule
        # lr_scheduleE.step(epoch=epoch)
        lr_scheduleG.step(epoch=epoch)
        lr_scheduleF.step(epoch=epoch)

        # Stats
        # if epoch % args.n_stats == 0:
        #     stats['loss_g'].append(loss_g.item())
        #     stats['loss_e'].append(loss_e.item())
        #     stats['en_neg'].append(en_neg.data.item())
        #     stats['en_pos'].append(en_pos.data.item())
        #     stats['grad_norm_g'].append(grad_norm_g)
        #     stats['grad_norm_e'].append(grad_norm_e)
        #     stats['z_g_grad_norm'].append(z_g_grad_norm.item())
        #     stats['z_e_grad_norm'].append(z_e_grad_norm.item())
        #     stats['z_e_k_grad_norm'].append(z_e_k_grad_norm.item())
        #     stats['fid'].append(fid)
        #     interval.append(epoch + 1)
        #     plot_stats(output_dir, stats, interval)

        # Metrics
        if epoch == args.n_epochs or epoch % args.n_metrics == 0:

            fid = get_fid(n=args.n_fid_samples)
            if fid < fid_best:
                fid_best = fid
            logger.info('fid={}'.format(fid))

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
        if epoch > 10 and loss_g > 300:
            logger.info('early exit condition 1: epoch > 10 and loss_g > 300')
            return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
            return

        # if epoch > 40 and fid > 100:
        #     logger.info('early exit condition 2: epoch > 40 and fid > 100')
        #     return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
        #     return

    return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
    logger.info('done')



##########################################################################################################
## Metrics

from fid_v2_tf_cpu import fid_score

def is_xsede():
    import socket
    return 'psc' in socket.gethostname()


def compute_fid(args, x_data, x_samples, use_cpu=False):

    assert type(x_data) == np.ndarray
    assert type(x_samples) == np.ndarray

    # RGB
    assert x_data.shape[3] == 3
    assert x_samples.shape[3] == 3

    # NHWC
    assert x_data.shape[1] == x_data.shape[2]
    assert x_samples.shape[1] == x_samples.shape[2]

    # [0,255]
    assert np.min(x_data) > 0.-1e-4
    assert np.max(x_data) < 255.+1e-4
    assert np.mean(x_data) > 10.

    # [0,255]
    assert np.min(x_samples) > 0.-1e-4
    assert np.max(x_samples) < 255.+1e-4
    assert np.mean(x_samples) > 1.

    if use_cpu:
        def create_session():
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.0
            config.gpu_options.visible_device_list = ''
            return tf.Session(config=config)
    else:
        def create_session():
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            config.gpu_options.visible_device_list = str(args.device)
            return tf.Session(config=config)

    path = '/tmp' if not is_xsede() else '/pylon5/ac561ep/enijkamp/inception'

    fid = fid_score(create_session, x_data, x_samples, path, cpu_only=use_cpu)

    return fid

def compute_fid_nchw(args, x_data, x_samples):

    to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))

    x_data_nhwc = to_nhwc(255 * x_data)
    x_samples_nhwc = to_nhwc(255 * x_samples)

    fid = compute_fid(args, x_data_nhwc, x_samples_nhwc)

    return fid


def test(args_job, output_dir, path_check_point):

    #################################################
    ## preamble

    args = parse_args()
    args = pygrid.overwrite_opt(args, args_job)
    args = to_named_dict(args)

    set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)

    # makedirs_exp(output_dir)

    job_id = int(args['job_id'])

    logger = setup_logging('job{}'.format(job_id), output_dir, console=True)
    #logger.info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')


    #################################################
    ## model

    netG = _netG(args)
    netF = _netF(args, nz=args.nz)


    ckp=torch.load(path_check_point)
    netG.load_state_dict(ckp['netG'])
    netF.load_state_dict(ckp['netF'])


    netG = netG.to(device)
    netF = netF.to(device)

    #logger.info(netG)
    #logger.info(netF)

    def eval_flag():
        netG.eval()
        netF.eval()


    #################################################
    ## test

    n = 50000

    logger.info('computing fid with {} samples'.format(n))


    eval_flag()
    to_range_0_1 = lambda x: (x + 1.) / 2.

    def sample_x():

        z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
        z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(int(z_sample.shape[0])).to(device),
                     reverse=True, return_obj=False)

        x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
        x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()

        return x_samples

    x_samples = torch.cat([sample_x() for _ in range(int(n / args.batch_size))]).numpy()

    ds_train, ds_test = get_dataset(args)
    ds_fid = np.array(torch.stack([to_range_0_1(ds_train[i][0]) for i in range(len(ds_train))]).cpu().numpy())

    fid = compute_fid_nchw(args, ds_fid, x_samples)

    logger.info('fid={}'.format(fid))



    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=True, num_workers=0)

    mse = nn.MSELoss(reduction='sum')

    def sample_langevin_post_z_with_flow(z, x, netG, netF, verbose=False):
        z = z.clone().detach()
        z.requires_grad = True

        for i in range(args.g_l_steps):
            x_hat = netG(z)
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat, x) #/ x.shape[0]
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            z1, logdet, _ = netF(torch.squeeze(z), objective=torch.zeros(int(z.shape[0])).to(device), init=False)
            prior_ll = -0.5 * (z1 ** 2)
            prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
            ll = prior_ll + logdet
            #f_log_lkhd = -ll.mean()
            f_log_lkhd = -ll.sum()



            #log_p, logdet, _ = netF(z)
            #logdet = logdet.mean()
            #f_log_lkhd, _, _ = calc_loss(log_p, logdet, args.f_in_channel, n_bins=2.0 ** args.n_bits)

            z_grad_f = torch.autograd.grad(f_log_lkhd, z)[0]

            z.data = z.data - 0.5 * args.g_l_step_size * args.g_l_step_size * (z_grad_g + z_grad_f)
            if args.g_l_with_noise:
                z.data += args.g_l_step_size * torch.randn_like(z).data

            z_grad_g_grad_norm = z_grad_g.view(args.batch_size, -1).norm(dim=1).mean()
            z_grad_f_grad_norm = z_grad_f.view(args.batch_size, -1).norm(dim=1).mean()

        if verbose:
            logger.info('Langevin posterior: MSE={:8.3f}, f_log_lkhd={:8.3f}, z_grad_g_grad_norm={:8.3f}, z_grad_f_grad_norm={:8.3f}'.format(g_log_lkhd.item(), f_log_lkhd.item(), z_grad_g_grad_norm, z_grad_f_grad_norm))


        return z.detach(), z_grad_g_grad_norm, z_grad_f_grad_norm


    recon_error = 0
    for i, (x, y) in enumerate(dataloader_test, 0):

        x = x.to(device)

        z_g_0 = torch.randn(x.shape[0], args.nz, 1, 1).to(device)
        z_g_k = sample_langevin_post_z_with_flow(z_g_0, x, netG, netF, verbose=False)[0]
        x_hat = netG(z_g_k.detach())
        recon_error = recon_error + float(mse(x_hat, x).cpu().data.numpy()) / x.shape[0] / 3 / args.img_size / args.img_size

    recon_error = recon_error / (i + 1)
    logger.info('reconstruction error={}'.format(recon_error))






##########################################################################################################
## Plots

import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_stats(output_dir, stats, interval):
    content = stats.keys()
    # f = plt.figure(figsize=(20, len(content) * 5))
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        axs[j].plot(interval, v)
        axs[j].set_ylabel(k)

    f.savefig(os.path.join(output_dir, 'stat.pdf'), bbox_inches='tight')
    f.savefig(os.path.join(output_dir, 'stat.png'), bbox_inches='tight')
    plt.close(f)



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


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    free_gpu = np.argmax(memory_available)
    print('set gpu', free_gpu, 'with', np.max(memory_available), 'mb')
    return free_gpu


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


def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d


##########################################################################################################
## Main

def makedirs_exp(output_dir):
    os.makedirs(output_dir + '/samples')
    os.makedirs(output_dir + '/ckpt')

def main():

    # print_gpus()

    fs_prefix = './' 

    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = pygrid.get_output_dir(exp_id, fs_prefix=fs_prefix)

    # run
    copy_source(__file__, output_dir)
    opt = {'job_id': int(0), 'status': 'open', 'device': get_free_gpu()}

    # training mode
    #train(opt, output_dir, output_dir, {})

    # testing mode
    path_check_point = '/home/kenny/extend/latent-space-flow-prior/output/train_svhn3/2021-07-24-03-48-20_fid=23.14/ckpt/ckpt_000160.pth'

    test(opt, output_dir, path_check_point)


if __name__ == '__main__':
    main()
