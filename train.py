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
import pytorch_fid_wrapper as pfw

import pygrid

from model import _netG, _netF, weights_init_xavier

##########################################################################################################
## Parameters

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_mode', action='store_true', default=False, help='test or not')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--dataset', type=str, default='svhn', choices=['svhn', 'cifar10', 'celeba_crop', 'celeba_hq256'])
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', default=3, type=int)
    parser.add_argument('--ngf', default=64, type=int, help='feature dimensions of generator')

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

    parser.add_argument('--g_lr', default=0.0004, type=float) # 0.0004
    parser.add_argument('--f_lr', default=0.0004, type=float) # 0.0004

    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--f_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')

    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--f_max_norm', type=float, default=100, help='max norm allowed')

    parser.add_argument('--g_decay',  default=0, type=float, help='weight decay for gen')
    parser.add_argument('--f_decay', default=0, type=float, help='weight decay for flow')

    parser.add_argument('--g_gamma', default=0.998, type=float, help='lr decay for gen')
    parser.add_argument('--f_gamma', default=0.998, type=float, help='lr decay for flow')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--f_beta1', default=0.5, type=float)
    parser.add_argument('--f_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=20, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')

    parser.add_argument('--n_ckpt', type=int, default=1, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=10, help='fid each n epochs')    #
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')
    parser.add_argument('--n_fid_samples', type=int, default=50000)

    parser.add_argument('--path_check_point', type=str, default=None)
    parser.add_argument('--testing_reconstruct', action='store_true', default=False, help='testing the reconstruction or not')


    return parser.parse_args()

def statistics(a):
    return "%.4f +- %.4f [%.4f-%.4f] : sum %.4f" % (a.mean(), a.std(), a.max(), a.min(), a.sum())

class Fid_calculator(object):

    def __init__(self, args, training_data):
        pfw.set_config(batch_size=args.batch_size, device="cuda")
        training_data = training_data.repeat(1,3 if training_data.shape[1] == 1 else 1,1,1)
        print("precalculate FID distribution for training data...")
        print("training data shape", training_data.shape, training_data.max(), training_data.min())
        self.real_m, self.real_s = pfw.get_stats(training_data)
        print("realm:", statistics(self.real_m), "reals:", statistics(self.real_s))

    def fid(self, data):
        data[torch.isnan(data)] = 0
        print("generated data shape", data.shape, data.max(), data.min())
        data[data > 1] = 1
        data[data < -1] = -1
        data = data.repeat(1,3 if data.shape[1] == 1 else 1,1,1)
        print("Before update realm:", statistics(self.real_m), "reals:", statistics(self.real_s))
        fid = pfw.fid(data, real_m=self.real_m.copy(), real_s=self.real_s.copy())
        print("After update realm:", statistics(self.real_m), "reals:", statistics(self.real_s))
        return fid

##########################################################################################################
## Data

def get_dataset(args):

    fs_prefix = './'

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

    elif args.dataset == 'cifar10':

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

    elif args.dataset == 'celeba':

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

    elif args.dataset == 'celeba_crop':

        crop = lambda x: transforms.functional.crop(x, 45, 25, 173-45, 153-25)

        import torchvision.transforms as transforms

        ds_train = torchvision.datasets.CelebA(fs_prefix + 'data', split='train', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        ds_val = torchvision.datasets.CelebA(fs_prefix + 'data', split='valid', download=True,
                                                    transform=transforms.Compose([
                                                        transforms.Lambda(crop),
                                                        transforms.Resize(args.img_size),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        return ds_train, ds_val

    elif args.dataset == 'celeba_hq256':

        import torchvision.transforms as transforms
        transform_train = transforms.Compose([transforms.Resize(args.img_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_val = transforms.Compose([transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        ds_train = torchvision.datasets.ImageFolder(root='./data/CelebAMask-HQ', transform=transform_train)
        ds_val = torchvision.datasets.ImageFolder(root='./data/CelebAMask-HQ', transform=transform_val)

        return ds_train, ds_val

    else:
        raise ValueError(args.dataset)

##########################################################################################################

def train(args, output_dir, path_check_point):

    #################################################
    ## preamble



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
    # ds_fid = np.array(torch.stack([to_range_0_1(ds_train[i][0]) for i in range(len(ds_train))]).cpu().numpy())
    ds_fid = torch.stack([to_range_0_1(ds_train[i][0]) for i in range(len(ds_train))]).cpu()
    print("training data:", statistics(ds_fid))
    fid_calculator = Fid_calculator(args, ds_fid)
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
            f_log_lkhd = -ll.sum()


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
    ## train

    # resume the training (1) for fine-tuning or (2) because of failure
    if path_check_point:
        ckp = torch.load(path_check_point)
        netG.load_state_dict(ckp['netG'])
        netF.load_state_dict(ckp['netF'])
        optG.load_state_dict(ckp['optG'])
        optF.load_state_dict(ckp['optF'])
        epoch_start=ckp['epoch']+1
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

            if args.f_is_grad_clamp:
                 torch.nn.utils.clip_grad_norm_(netF.parameters(), args.f_max_norm)
            optF.step()


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

        # Metrics
        if epoch == args.n_epochs or epoch % args.n_metrics == 0:

            try:
                eval_flag()

                def sample_x():
                    z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
                    z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(int(z_sample.shape[0])).to(device),
                                 reverse=True, return_obj=False)
                    x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
                    x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()
                    return x_samples

                x_samples = torch.cat([sample_x() for _ in range(int(args.n_fid_samples / args.batch_size))])
                plot('{}/samples/{:>06d}_x_z_test_fid.png'.format(output_dir, epoch), (x_samples[:100] - 0.5) * 2.0)
                fid = fid_calculator.fid(x_samples)

            except Exception as e:
                print(e)
                logger.critical(e, exc_info=True)
                logger.info('FID failed')
                fid = 10000

            if fid < fid_best:
                fid_best = fid
            logger.info('fid={}'.format(fid))

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

    #return_dict['stats'] = {'fid_best': fid_best, 'fid': fid, 'mse': loss_g.data.item()}
    logger.info('done')



##########################################################################################################
## Metrics

# from fid_v2_tf_cpu import fid_score

def is_xsede():
    import socket
    return 'psc' in socket.gethostname()

#################################################
## test

def test(args, output_dir, path_check_point):

    #################################################
    ## preamble

    set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)

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

    def eval_flag():
        netG.eval()
        netF.eval()


    #################################################
    ## test


    logger.info('computing fid with {} samples'.format(args.n_fid_samples))

    eval_flag()
    to_range_0_1 = lambda x: (x + 1.) / 2.

    def sample_x():

        z_sample = torch.randn(args.batch_size, args.nz, 1, 1).to(device)
        z_f_k = netF(torch.squeeze(z_sample), objective=torch.zeros(int(z_sample.shape[0])).to(device),
                     reverse=True, return_obj=False)

        x_samples = netG(torch.reshape(z_f_k, (z_f_k.shape[0], z_f_k.shape[1], 1, 1)))
        x_samples = to_range_0_1(x_samples).clamp(min=0., max=1.).detach().cpu()

        return x_samples

    ## save a batch of synthesized examples
    def plot(p, x):
        return torchvision.utils.save_image(torch.clamp(x, -1., 1.), p, normalize=True, nrow=int(np.sqrt(args.batch_size)))

    x_saved = sample_x()
    plot('{}/synthesis.png'.format(output_dir), x_saved)


    x_samples = torch.cat([sample_x() for _ in range(int(args.n_fid_samples / args.batch_size))])


    ds_train, ds_test = get_dataset(args)
    assert len(ds_train) >= args.n_fid_samples
    ds_fid = torch.stack([to_range_0_1(ds_train[i][0]) for i in range(len(ds_train))]).cpu()
    fid_calculator = Fid_calculator(args, ds_fid)

    fid = fid_calculator.fid(x_samples)
    logger.info('fid={}'.format(fid))


    dataloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=True, num_workers=0)

    mse = nn.MSELoss(reduction='sum')

    def sample_langevin_post_z_with_flow(z, x, netG, netF, verbose=False):
        z = z.clone().detach()
        z.requires_grad = True

        g_l_steps_testing = args.g_l_steps * 20
        g_l_step_size_testing = args.g_l_step_size

        for i in range(g_l_steps_testing):
            x_hat = netG(z)
            g_log_lkhd = 1.0 / (2.0 * args.g_llhd_sigma * args.g_llhd_sigma) * mse(x_hat, x) #/ x.shape[0]
            z_grad_g = torch.autograd.grad(g_log_lkhd, z)[0]

            z1, logdet, _ = netF(torch.squeeze(z), objective=torch.zeros(int(z.shape[0])).to(device), init=False)
            prior_ll = -0.5 * (z1 ** 2)
            prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
            ll = prior_ll + logdet
            #f_log_lkhd = -ll.mean()
            f_log_lkhd = -ll.sum()

            z_grad_f = torch.autograd.grad(f_log_lkhd, z)[0]

            z.data = z.data - 0.5 * g_l_step_size_testing * g_l_step_size_testing * (z_grad_g + z_grad_f)
            #if args.g_l_with_noise:
            #    z.data += args.g_l_step_size * torch.randn_like(z).data

            z_grad_g_grad_norm = z_grad_g.view(args.batch_size, -1).norm(dim=1).mean()
            z_grad_f_grad_norm = z_grad_f.view(args.batch_size, -1).norm(dim=1).mean()

        if verbose:
            logger.info('Langevin posterior: MSE={:8.3f}, f_log_lkhd={:8.3f}, z_grad_g_grad_norm={:8.3f}, z_grad_f_grad_norm={:8.3f}'.format(g_log_lkhd.item(), f_log_lkhd.item(), z_grad_g_grad_norm, z_grad_f_grad_norm))


        return z.detach(), z_grad_g_grad_norm, z_grad_f_grad_norm

    if args.testing_reconstruct:

        recon_error = 0
        for i, (x, y) in enumerate(dataloader_test, 0):

            x = x.to(device)

            z_g_0 = torch.randn(x.shape[0], args.nz, 1, 1).to(device)
            z_g_k = sample_langevin_post_z_with_flow(z_g_0, x, netG, netF, verbose=False)[0]
            x_hat = netG(z_g_k.detach())
            # x_hat = to_range_0_1(x_hat).clamp(min=0., max=1.)
            recon_error = recon_error + float(mse(x_hat, x).cpu().data.numpy()) / x.shape[0] / 3 / args.img_size / args.img_size

            if i==0:
                plot('{}/reconstrction.png'.format(output_dir), x_hat)
                plot('{}/original.png'.format(output_dir), x)


        recon_error = recon_error / (i + 1)
        logger.info('reconstruction error={}'.format(recon_error))


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

    output = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader').readlines()
    memory_available = [int(line.split()[0]) for line in output]
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

    args = parse_args()
    args = pygrid.overwrite_opt(args, opt)
    args = to_named_dict(args)


    if args.test_mode:
        # testing mode
        test(args, output_dir, args.path_check_point)

    else:
        # training mode
        train(args, output_dir, args.path_check_point)

if __name__ == '__main__':
    main()
