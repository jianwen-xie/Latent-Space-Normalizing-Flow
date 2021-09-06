import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn
import numpy as np
from scipy import linalg as la
from math import log, pi, exp
##########################################################################################################
## Model

#######################################
## generator

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)

def get_activation(name, args):
    return {'gelu': GELU(), 'lrelu': nn.LeakyReLU(args.g_activation_leak), 'mish': Mish(), 'swish': Swish()}[name]


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, args):
        super().__init__()

        f = get_activation(args.g_activation, args)

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(args.nz, args.ngf*8, 4, 1, 0, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.ngf*8) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*8, args.ngf*4, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.ngf*4) if args.g_batchnorm else nn.Identity(),
            f,

            nn.ConvTranspose2d(args.ngf*4, args.ngf*2, 4, 2, 1, bias = not args.g_batchnorm),
            nn.BatchNorm2d(args.ngf*2) if args.g_batchnorm else nn.Identity(),
            f,

            # if the image size is of 64 x 64, uncomment this layer
            #nn.ConvTranspose2d(args.ngf*2, args.ngf*1, 4, 2, 1, bias = not args.g_batchnorm),
            #nn.BatchNorm2d(args.ngf*1) if args.g_batchnorm else nn.Identity(),
            #f,

            nn.ConvTranspose2d(args.ngf*2, args.nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.gen(z)

####################################################
## EBM

class _netE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        apply_sn = sn if args.e_sn else lambda x: x

        f = get_activation(args.e_activation, args)

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(args.nz, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.ndf)),
            f,

            apply_sn(nn.Linear(args.ndf, args.nez))
        )

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, self.args.nez, 1, 1)

####################################################
## flow


################################### flow model ##################################
def int_shape(x):
    return list(map(int, x.shape))

def reverse_features(h, reverse=False):
    return h[:, :, :, ::-1]

class invertible_1x1_conv(nn.Module):
    def __init__(self, width, nz, pad="SAME", *args, **kwargs):
        super(invertible_1x1_conv, self).__init__(*args, **kwargs)
        w_shape = [nz, nz]
        # Sample a random orthogonal matrix:
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        self.w = nn.Parameter(torch.tensor(w_init, dtype=torch.float))

    def forward(self, z, logdet, reverse=False):
        w = self.w
        # dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
        dlogdet = torch.log(abs(torch.det(w.double()))).float()

        if not reverse:

            _w = w
            z = torch.matmul(z, _w)
            # z = tf.nn.conv2d(z, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
            logdet = logdet + dlogdet

            return z, logdet
        else:
            _w = torch.inverse(w)
            z = torch.matmul(z, _w)
            # z = tf.nn.conv2d(z, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
            logdet = logdet - dlogdet

            return z, logdet

class shuffle_features(nn.Module):
    def __init__(self, nz, *args, **kwargs):
        super(shuffle_features, self).__init__(*args, **kwargs)
        rng = np.random.RandomState(np.random.rand(0, 1000000))
        n_channels = nz
        self.indices = list(range(n_channels))
        rng.shuffle(self.indices)
        # Reverse it
        indices_inverse = [0] * n_channels
        for i in range(n_channels):
            indices_inverse[self.indices[i]] = i
        self.indices = nn.Parameter(torch.tensor(self.indices, dtype=torch.int), requires_grad=False)
        self.indices_inverse = nn.Parameter(torch.tensor(self.indices_inverse, dtype=torch.int), requires_grad=False)

    def forward(self, h, return_indices=False, reverse=False):
        _indices = self.indices
        if reverse:
            _indices = self.indices_reverse

        h = h.permute(1, 0)
        h = torch.gather(h, dim=0, index=_indices)
        h = h.permute(1, 0)

        if return_indices:
            return h, self.indices
        return h

class actnorm(nn.Module):
    def __init__(self, nz, *args, **kwargs):
        super(actnorm, self).__init__(*args, **kwargs)
        self.b = nn.Parameter(torch.randn(1, nz) * 0.05)
        self.register_parameter(name='bias', param=self.b)
        _shape = (1,nz)
        self.logs = nn.Parameter(torch.randn(*_shape) * 0.05)

    def actnorm_center(self, x, reverse=False, init=False):
        shape = x.shape
        assert len(shape) == 2
        x_mean = torch.mean(x, dim=0, keepdims=True)
        if init:
            initial_value = - x_mean
            self.b.data.copy_(initial_value.data)

        if not reverse:
            x = x + self.b
        else:
            x = x - self.b

        return x

    def actnorm_scale(self, x, scale=1., logdet=None, logscale_factor=3., batch_variance=False, reverse=False, init=False):
        shape = x.shape
        assert len(shape) == 2
        x_var = torch.mean(x ** 2, dim=0, keepdims=True)
        logdet_factor = 1
        _shape = (1, int_shape(x)[1])

        if batch_variance:
            x_var = torch.mean(x ** 2, keepdims=True)

        if init:
            initial_value = torch.log(scale / (torch.sqrt(x_var) + 1e-6)) / logscale_factor
            self.logs.data.copy_(initial_value.data)

        logs = self.logs * logscale_factor
        #print(logs.shape, logs)

        if not reverse:
            x = x * torch.exp(logs)
        else:
            x = x * torch.exp(-logs)

        if logdet != None:
            dlogdet = torch.sum(logs) * logdet_factor
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x

    def forward(self, x, reverse=False, batch_variance=False, scale=1., logdet=None, logscale_factor=3., init=False):

        if not reverse:
            x = self.actnorm_center(x, reverse, init=init)
            x = self.actnorm_scale(x, scale, logdet, logscale_factor, batch_variance, reverse, init=init)
            if logdet != None:
                x, logdet = x
        else:
            x = self.actnorm_scale(x, scale, logdet, logscale_factor, batch_variance, reverse, init=init)
            if logdet != None:
                x, logdet = x
            x = self.actnorm_center(x, reverse, init=init)
        if logdet != None:
            return x, logdet
        return x

class f(nn.Module):
    # f is used to generate the mean and scale in the transformation (get input from the direct copy branch)
    def __init__(self, width, n_in=None, n_out=None, name='f', *args, **kwargs):
        super(f, self).__init__(*args, **kwargs)
        self.n_out = n_out
        self.fc_1 = fc(n_in, width)
        self.fc_2 = fc(width, width)
        self.fc_zeros = fc_zeros(width, n_out)


    def forward(self, h, init):
        h = F.relu(self.fc_1(h, init=init))
        h = F.relu(self.fc_2(h, init=init))
        h = self.fc_zeros(h)
        return h

class fc(nn.Module):
    def __init__(self, n_in, width, edge_bias=True, *args, **kwargs):
        super(fc, self).__init__(*args, **kwargs)
        self.width = width
        self.edge_bias = edge_bias
        self.actnorm = actnorm(nz=width)
        self.w = nn.Parameter(torch.randn(n_in, self.width) * 0.05)
        self.b = nn.Parameter(torch.zeros(1, self.width))

    def forward(self, x, do_weightnorm=False, do_actnorm=True, context1d=None, skip=1, init=False):
        assert len(x.shape) == 2
        w = self.w
        if do_weightnorm:
            w = F.normalize(w, dim=0)
        x = torch.matmul(x, w)
        if do_actnorm:
            x = self.actnorm(x, init=init)
        else:
            x = x + self.b

        return x

class fc_zeros(nn.Module):
    def __init__(self, n_in, width, logscale_factor=3, edge_bias=True, *args, **kwargs):
        super(fc_zeros, self).__init__(*args, **kwargs)
        self.width = width
        self.logscale_factor = logscale_factor
        self.edge_bias = edge_bias
        self.w = nn.Parameter(torch.zeros(n_in, self.width))
        self.b = nn.Parameter(torch.zeros(1, self.width))
        self.logs = nn.Parameter(torch.zeros(1, self.width))

    def forward(self, x, skip=1):
        # for the first call, initialize the weight
        w = self.w
        x = torch.matmul(x, w)
        x = x + self.b
        x = x * torch.exp(self.logs * self.logscale_factor)
        return x

class revnet2d(nn.Module):
    def __init__(self, hps, nz, *args, **kwargs):
        super(revnet2d, self).__init__(*args, **kwargs)
        self.hps = hps
        self.revnet2d_step_s = nn.ModuleList([revnet2d_step(str(i), hps, nz=nz) for i in range(hps.f_depth)])
    def forward(self, z, logdet, reverse=False, init=False):
        if not reverse:
            for i in range(self.hps.f_depth):
                z, logdet = self.revnet2d_step_s[i](z, logdet, reverse, init=init)
        else:
            for i in reversed(range(self.hps.f_depth)):
                z, logdet = self.revnet2d_step_s[i](z, logdet, reverse, init=init)

        return z, logdet

class revnet2d_step(nn.Module):
    def __init__(self, id, hps, nz, *args, **kwargs):
        super(revnet2d_step, self).__init__(*args, **kwargs)
        self.actnorm = actnorm(nz=nz)
        self.hps = hps
        if self.hps.f_flow_permutation == 1:
            self.shuffle_features = shuffle_features(nz=nz)
            self.invertible_1x1_conv = None
        elif self.hps.f_flow_permutation == 2:
            self.invertible_1x1_conv = invertible_1x1_conv(hps.f_width, nz=nz)
            self.shuffle_features = None
        else:
            raise Exception()

        self.id = id

        assert nz % 2 == 0
        if self.hps.f_flow_coupling == 0:
            self.f = f(self.hps.f_width, nz//2, nz//2, name='f_' + self.id)
        elif self.hps.f_flow_coupling == 1:
            self.f = f(self.hps.f_width, nz//2, nz, name='f_' + self.id)

    def forward(self, z, logdet, reverse, init):
        n_z = int_shape(z)[-1]
        if not reverse:
            z, logdet = self.actnorm(z, logdet=logdet, init=init)
            #print(logdet, logdet.shape)

            if self.hps.f_flow_permutation == 0:
                z = reverse_features(z)
            elif self.hps.f_flow_permutation == 1:
                z = self.shuffle_features(z)
            elif self.hps.f_flow_permutation == 2:
                z, logdet = self.invertible_1x1_conv(z, logdet=logdet)
            else:
                raise Exception()

            z1 = z[:, :n_z // 2]
            z2 = z[:, n_z // 2:]

            if self.hps.f_flow_coupling == 0:
                z2 = z2 + self.f(z1, init=init)
            elif self.hps.f_flow_coupling == 1:
                h = self.f(z1, init=init)
                shift = h[:, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = torch.sigmoid(h[:, 1::2] + 2.)
                z2 = z2 + shift
                z2 = z2 * scale
                # print(logdet)
                # print(scale)
                logdet = logdet + torch.sum(torch.log(scale), dim=1)
            else:
                raise Exception()

            z = torch.cat([z1, z2], 1)

        else:

            z1 = z[:, :n_z // 2]
            z2 = z[:, n_z // 2:]

            if self.hps.f_flow_coupling == 0:
                z2 -= self.f(z1, init=init)
            elif self.hps.f_flow_coupling == 1:
                h = self.f(z1, init=init)
                shift = h[:, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = torch.sigmoid(h[:, 1::2] + 2.)
                z2 /= scale
                z2 -= shift
                logdet -= torch.sum(torch.log(scale), dim=1)
            else:
                raise Exception()

            # print('z1', z1.shape)
            # print('z2', z2.shape)

            z = torch.cat([z1, z2], 1)

            if self.hps.f_flow_permutation == 0:
                z = reverse_features(z, reverse=True)
            elif self.hps.f_flow_permutation == 1:
                z = self.shuffle_features(z, reverse=True)
            elif self.hps.f_flow_permutation == 2:
                z, logdet = self.invertible_1x1_conv(z, logdet, reverse=True)
            else:
                raise Exception()

            z, logdet = self.actnorm(z, logdet=logdet, reverse=True, init=init)

        return z, logdet

class _netF(nn.Module):
    def __init__(self, hps, nz,  *args, **kwargs):
        super(_netF, self).__init__(*args, **kwargs)
        revnet2d_s = []
        self.hps = hps
        for i in range(hps.f_n_levels):
            revnet2d_s.append(revnet2d(hps, nz=nz))
            if i < hps.f_n_levels - 1:
                # self.split2d_s.append(split2d(hps)
                # should implement the split layer in the future
                raise NotImplementedError
        self.revnet2d_s = nn.ModuleList(revnet2d_s)

    def forward(self, z, objective, init=False, reverse=False, eps=None, eps_std=None, z2_s=None, return_obj=False):
        if not reverse:
            eps_forward = []
            for i in range(self.hps.f_n_levels):
                # _print('codec->z_{}'.format(i), np.array(z[0][0][0]))
                z, objective = self.revnet2d_s[i](z, objective, init=init)
                if i < self.hps.f_n_levels - 1:
                    # _print('codec->split2d', np.array(z[0][0][0]))
                    z, objective, _eps = self.split2d_s[i](z, objective=objective)
                    eps_forward.append(_eps)
            return z, objective, eps_forward
        else:
            eps = eps if eps else [None] * self.hps.f_n_levels
            for i in reversed(range(self.hps.f_n_levels)):
                # print(i, 'z', z.shape)
                if i < self.hps.f_n_levels - 1:
                    # z, objective = self.split2d_reverse_s[i](z, objective=objective, eps=eps[i], eps_std=eps_std, z2=z2_s[i] if z2_s else None)
                    z, objective = self.split2d_s[i](z, reverse=True, objective=objective, eps=eps[i], eps_std=eps_std,
                                                     z2=z2_s[i] if z2_s else None)

                z, objective = self.revnet2d_s[i](z, objective, reverse=True)
                # print(i, 'z', z.shape)
            if not return_obj:
                return z
            else:
                return z, -objective


class _netF2(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = args.f_in_channel
        for i in range(args.f_n_block - 1):
            #self.blocks.append(Block(n_channel, args.f_n_flow, affine=args.f_affine, conv_lu=args.f_conv_lu))
            self.blocks.append(Block(n_channel, args.f_n_flow, affine=args.f_affine))
            #n_channel *= 2
        self.blocks.append(Block(n_channel, args.f_n_flow, split=False, affine=args.f_affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)

            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)

        return input

