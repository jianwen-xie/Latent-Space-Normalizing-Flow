import os
import sys
import time
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import random
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as sched

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
        self.revnet2d_step_s = nn.ModuleList([revnet2d_step(str(i), hps, nz=nz) for i in range(hps.depth)])
    def forward(self, z, logdet, reverse=False, init=False):
        if not reverse:
            for i in range(self.hps.depth):
                z, logdet = self.revnet2d_step_s[i](z, logdet, reverse, init=init)
        else:
            for i in reversed(range(self.hps.depth)):
                z, logdet = self.revnet2d_step_s[i](z, logdet, reverse, init=init)

        return z, logdet

class revnet2d_step(nn.Module):
    def __init__(self, id, hps, nz, *args, **kwargs):
        super(revnet2d_step, self).__init__(*args, **kwargs)
        self.actnorm = actnorm(nz=nz)
        self.hps = hps
        if self.hps.flow_permutation == 1:
            self.shuffle_features = shuffle_features(nz=nz)
            self.invertible_1x1_conv = None
        elif self.hps.flow_permutation == 2:
            self.invertible_1x1_conv = invertible_1x1_conv(hps.width, nz=nz)
            self.shuffle_features = None
        else:
            raise Exception()

        self.id = id

        assert nz % 2 == 0
        if self.hps.flow_coupling == 0:
            self.f = f(self.hps.width, nz//2, nz//2, name='f_' + self.id)
        elif self.hps.flow_coupling == 1:
            self.f = f(self.hps.width, nz//2, nz, name='f_' + self.id)

    def forward(self, z, logdet, reverse, init):
        n_z = int_shape(z)[-1]
        if not reverse:
            z, logdet = self.actnorm(z, logdet=logdet, init=init)
            #print(logdet, logdet.shape)

            if self.hps.flow_permutation == 0:
                z = reverse_features(z)
            elif self.hps.flow_permutation == 1:
                z = self.shuffle_features(z)
            elif self.hps.flow_permutation == 2:
                z, logdet = self.invertible_1x1_conv(z, logdet=logdet)
            else:
                raise Exception()

            z1 = z[:, :n_z // 2]
            z2 = z[:, n_z // 2:]

            if self.hps.flow_coupling == 0:
                z2 = z2 + self.f(z1, init=init)
            elif self.hps.flow_coupling == 1:
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

            if self.hps.flow_coupling == 0:
                z2 -= self.f(z1, init=init)
            elif self.hps.flow_coupling == 1:
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

            if self.hps.flow_permutation == 0:
                z = reverse_features(z, reverse=True)
            elif self.hps.flow_permutation == 1:
                z = self.shuffle_features(z, reverse=True)
            elif self.hps.flow_permutation == 2:
                z, logdet = self.invertible_1x1_conv(z, logdet, reverse=True)
            else:
                raise Exception()

            z, logdet = self.actnorm(z, logdet=logdet, reverse=True, init=init)

        return z, logdet

class flow(nn.Module):
    def __init__(self, hps, nz,  *args, **kwargs):
        super(flow, self).__init__(*args, **kwargs)
        revnet2d_s = []
        self.hps = hps
        for i in range(hps.n_levels):
            revnet2d_s.append(revnet2d(hps, nz=nz))
            if i < hps.n_levels - 1:
                # self.split2d_s.append(split2d(hps)
                # should implement the split layer in the future
                raise NotImplementedError
        self.revnet2d_s = nn.ModuleList(revnet2d_s)

    def forward(self, z, objective, init=False, reverse=False, eps=None, eps_std=None, z2_s=None, return_obj=False):
        if not reverse:
            eps_forward = []
            for i in range(self.hps.n_levels):
                # _print('codec->z_{}'.format(i), np.array(z[0][0][0]))
                z, objective = self.revnet2d_s[i](z, objective, init=init)
                if i < self.hps.n_levels - 1:
                    # _print('codec->split2d', np.array(z[0][0][0]))
                    z, objective, _eps = self.split2d_s[i](z, objective=objective)
                    eps_forward.append(_eps)
            return z, objective, eps_forward
        else:
            eps = eps if eps else [None] * self.hps.n_levels
            for i in reversed(range(self.hps.n_levels)):
                # print(i, 'z', z.shape)
                if i < self.hps.n_levels - 1:
                    # z, objective = self.split2d_reverse_s[i](z, objective=objective, eps=eps[i], eps_std=eps_std, z2=z2_s[i] if z2_s else None)
                    z, objective = self.split2d_s[i](z, reverse=True, objective=objective, eps=eps[i], eps_std=eps_std,
                                                     z2=z2_s[i] if z2_s else None)

                z, objective = self.revnet2d_s[i](z, objective, reverse=True)
                # print(i, 'z', z.shape)
            if not return_obj:
                return z
            else:
                return z, -objective

################################### ebm model ##############################
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
    return {'gelu': GELU(), 'lrelu': nn.LeakyReLU(0.2), 'mish': Mish(), 'swish': Swish()}[name]

def ebm_sample(x_flow, ebm_model, hps):
    x_var = torch.autograd.Variable(x_flow, requires_grad=True)
    for k in range(hps.num_step):
        net_prime = torch.autograd.grad(ebm_model(x_var).sum(), [x_var])[0]
        delta = 0.5 * hps.step_size * hps.step_size * ((x_var / hps.sigma ** 2) - net_prime) \
            + hps.step_size * (1.0 - 1.0 / (hps.num_step - k)) * torch.randn_like(x_var).data
        x_var.data -= delta
    return x_var

class ebm(nn.Module):
    def __init__(self, hps, nz, *args, **kwargs):
        super(ebm, self).__init__(*args, **kwargs)
        self.act = get_activation(hps.ebm_act, args)
        self.model = nn.Sequential(
            nn.Linear(nz, 128),
            self.act,
            nn.Linear(128, 128),
            self.act,
            nn.Linear(128, 128),
            self.act,
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x).sum(1, keepdims=True)

################################## generate data ###########################
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32")

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] = point[0] + center[0]
            point[1] = point[1] + center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] = features[:, 0] + 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((d1x, d1y)))) / 3
        x = x + np.random.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        return inf_train_gen("8gaussians", rng, batch_size)

################################### train ##################################
def train(hps, logdir):
    # initialize flow, we need first initialize the model to create a series of parameters
    x = torch.tensor(inf_train_gen(hps.problem, batch_size=1024), dtype=torch.float32)
    x = x
    flow_model = flow(hps, nz=x.shape[-1])
    ebm_model = ebm(hps, nz=x.shape[-1])

    flow_model(x, objective=torch.zeros(int(x.shape[0])), init=True)
    ebm_model(x)

    device = 'cuda' if torch.cuda.is_available() and hps.gpu_ids else 'cpu'
    flow_model.to(device)
    ebm_model.to(device)
    print('flow parameters')
    for name, param in flow_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    print('ebm parameters')
    for name, param in ebm_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    flow_optimizer = optim.Adam(flow_model.parameters(), lr=hps.lr_flow)
    ebm_optimizer = optim.Adam(ebm_model.parameters(), lr=hps.lr_ebm)

    warm_up = hps.epoch_warmup * hps.n_batch_train
    flow_scheduler = sched.LambdaLR(flow_optimizer, lambda s: min(1., s / warm_up))
    ebm_scheduler = sched.LambdaLR(ebm_optimizer, lambda s: min(1., s / warm_up))
    start_epoch = 0
    if hps.resume:
        print('Resuming from checkpoint at {}...'.format(hps.restore_path))
        assert os.path.exists(hps.restore_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(hps.restore_path)
        flow_model.load_state_dict(checkpoint['flow_model'])
        ebm_model.load_state_dict(checkpoint['ebm_model'])
        start_epoch = checkpoint['epoch'] + 1

    start_time = time.time()
    for i in range(start_epoch, hps.epochs):
        flow_model.train()
        ebm_model.train()

        x = torch.tensor(inf_train_gen(hps.problem, batch_size=hps.n_batch_train), dtype=torch.float32)
        x = x.to(device)

        z = torch.randn(hps.n_batch_train, 2).to(device)
        x_flow = flow_model(z, objective=torch.zeros(int(x.shape[0])).to(device), reverse=True, return_obj=False)

        #x_init = hps.sigma * torch.randn(hps.n_batch_train, 2).to(device)
        x_ebm = ebm_sample(x_flow.clone(), ebm_model, hps).detach()

        mse_loss = torch.sum(torch.mean((x_ebm - x_flow) ** 2, dim=0))

        # update flow
        for _ in range(1):
            flow_optimizer.zero_grad()
            z, logdet, _ = flow_model(x_ebm, objective=torch.zeros(int(x.shape[0])).to(device), init=False)
            #z, logdet, _ = flow_model(x, objective=torch.zeros(int(x.shape[0])).to(device), init=False)
            prior_ll = -0.5 * (z ** 2)
            prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
            ll = prior_ll + logdet
            nll = -ll.mean()
            nll.backward()
            flow_optimizer.step()
            flow_scheduler.step()

        # update ebm
        for _ in range(1):
            with torch.no_grad():
                x = torch.tensor(inf_train_gen(hps.problem, batch_size=hps.n_batch_train), dtype=torch.float32)
                x = x.to(device)
                #z = torch.randn(hps.n_batch_train, 2).to(device)
                #x_flow = flow_model(z, objective=torch.zeros(int(x.shape[0])).to(device), reverse=True,return_obj=False).detach()
            #x_ebm = ebm_sample(x_flow.clone(), ebm_model, hps).detach()


            ebm_optimizer.zero_grad()
            en_pos = ebm_model(x).mean()
            en_neg = ebm_model(x_ebm).mean()
            ebm_loss = en_neg - en_pos

            #z_pos, logdet_pos, _ = flow_model(x, objective=torch.zeros(int(x.shape[0])).to(device), init=False)
            #z_neg, logdet_neg, _ = flow_model(x_ebm, objective=torch.zeros(int(x_ebm.shape[0])).to(device), init=False)
            ebm_loss.backward()
            ebm_optimizer.step()
            ebm_scheduler.step()
            if ebm_loss < 0:
                break

        if torch.isnan(en_pos) or torch.isnan(en_neg) or torch.isnan(mse_loss) or torch.isnan(nll):
            print('Blow up')
            return

        if i % hps.log_iter == 0:
            print('Epoch {}, time {:.4f}, mse loss {:.4f}, flow loss {:.4f}, pos en {:.4f}, neg en {:.4f}, ebm loss {:.4f}'\
                  .format(i, time.time() - start_time, mse_loss, nll, en_pos, en_neg, ebm_loss))


        if i % hps.image_iter == 0:
            LOW, HIGH = -4, 4
            nps = 500
            side = np.linspace(LOW, HIGH, nps)
            xx, yy = np.meshgrid(side, side)
            x_grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
            x_grid = torch.tensor(x_grid, dtype=torch.float).to(device)
            x = x.detach().cpu().numpy()
            x_flow = x_flow.detach().cpu().numpy()
            x_ebm = x_ebm.detach().cpu().numpy()

            en_grid = ebm_model(x_grid) - 0.5 * torch.sum((x_grid / hps.sigma) ** 2, dim=1, keepdims=True)

            #print(en_grid.shape)
            #print(.shape)
            #en_grid /= en_grid.sum()
            en_grid = en_grid.detach().cpu().numpy()
            en_grid = np.reshape(en_grid, (nps, nps))
            en_grid = np.flip(en_grid, axis=0)

            en_prob = np.exp(en_grid)
            en_prob /= np.sum(en_prob)

            z, logdet, _ = flow_model(x_grid, objective=torch.zeros(int(x_grid.shape[0])).to(device), init=False)
            prior_ll = -0.5 * (z ** 2 )
            prior_ll = prior_ll.flatten(1).sum(-1) + np.log(2 * np.pi)
            ll = prior_ll + logdet
            flow_prob = torch.exp(ll)
            flow_prob = flow_prob.detach().cpu().numpy()
            flow_prob = np.reshape(flow_prob, (nps, nps))
            flow_prob = np.flip(flow_prob, axis=0)

            fig, axes = plt.subplots(2, 3)
            axes[0, 0].scatter(x[:, 0], x[:, 1])
            axes[0, 0].set_xlim([-3.5, 3.5])
            axes[0, 0].set_ylim([-3.5, 3.5])
            axes[0, 0].set_title('Observed sample')
            axes[0, 1].scatter(x_flow[:, 0], x_flow[:, 1])
            axes[0, 1].set_xlim([-3.5, 3.5])
            axes[0, 1].set_ylim([-3.5, 3.5])
            axes[0, 1].set_title('Flow sample')
            axes[0, 2].scatter(x_ebm[:, 0], x_ebm[:, 1])
            axes[0, 2].set_xlim([-3.5, 3.5])
            axes[0, 2].set_ylim([-3.5, 3.5])
            axes[0, 2].set_title('EBM sample')

            axes[1, 0].matshow(flow_prob)
            axes[1, 0].set_title('flow prob')
            axes[1, 1].matshow(en_grid)
            axes[1, 1].set_title('ebm energy')
            axes[1, 2].matshow(en_prob)
            axes[1, 2].set_title('ebm prob')


            fig.savefig(os.path.join(logdir, '{}.png').format(i))
            plt.close(fig)

        if i % hps.save_iter == 0:
            print('Saving ... ')
            state = {
                'ebm_model': ebm_model.state_dict(),
                'flow_model': flow_model.state_dict(),
                'epoch': i,
            }
            ckpt_dir = os.path.join(logdir, 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(state, os.path.join(ckpt_dir, '{}.pth.tar'.format(i)))


def main(hps):
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    np.set_printoptions(threshold=1, precision=4, linewidth=np.inf)

    if not hps.logdir:
        import datetime
        hps.logdir = os.path.join(os.path.splitext(os.path.basename(__file__))[0], hps.problem,
                                  datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    # Init
    logdir = os.path.join(hps.logdir, hps.problem + '_ebm500step')
    os.makedirs(logdir, exist_ok=True)

    # Set seeds
    np.random.seed(hps.seed)
    torch.manual_seed(hps.seed)
    random.seed(hps.seed)

    # Train model
    train(hps, logdir)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow++ on CelebA')
    def str2bool(s):
        return s.lower().startswith('t')
    parser.add_argument('--logdir', default='save', type=str, help='')
    parser.add_argument('--problem', default='2spirals', type=str, help='')
    parser.add_argument('--n_batch_train', default=512, type=int, help='')
    parser.add_argument('--n_batch_test', default=50, type=int, help='')
    parser.add_argument('--n_batch_init', default=256, type=int, help='')
    parser.add_argument('--lr_flow', default=0.00005, type=float, help='')
    parser.add_argument('--lr_ebm', default=0.0005, type=float, help='')
    parser.add_argument('--beta1', default=0.9, type=float, help='')
    parser.add_argument('--epochs', default=20001, type=int, help='')
    parser.add_argument('--epoch_warmup', default=10, type=int, help='')
    parser.add_argument('--width', default=64, type=int, help='')
    parser.add_argument('--depth', default=10, type=int, help='')
    parser.add_argument('--n_levels', default=1, type=int, help='')
    parser.add_argument('--rho', default=0.5, type=float, help='')
    parser.add_argument('--n_sample', default=1, type=int, help='')
    parser.add_argument('--learntop', default=True, type=str2bool, help='')
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--flow_permutation', default=2, type=int, help='')
    parser.add_argument('--flow_coupling', default=1, type=int, help='')
    parser.add_argument('--resume', default=False, type=str2bool, help='')
    parser.add_argument('--restore_path', default=None, type=str, help='')
    parser.add_argument('--gpu_ids', default=[1], type=eval, help='')
    parser.add_argument('--num_step', default=500, type=int, help='')
    parser.add_argument('--step_size', default=0.01, type=float, help='')
    parser.add_argument('--sigma', default=1.0, type=float, help='')
    parser.add_argument('--log_iter', default=5, type=int, help='')
    parser.add_argument('--image_iter', default=25, type=int, help='')
    parser.add_argument('--save_iter', default=1000, type=int, help='')
    parser.add_argument('--ebm_act', default='swish', type=str, help='')
    main(parser.parse_args())




