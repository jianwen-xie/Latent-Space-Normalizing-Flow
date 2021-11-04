import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import utils

from scipy import io as sio
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

from OT_dual import OT_solver, correspondence_reconstruction

import logging
import os

class decoder(nn.Module):
    def __init__(self, dim_z=20, dim_c=1, width=28, height=28):
        super(decoder, self).__init__()
        self.dim_c = dim_c
        self.dim_z = dim_z
        self.width = width
        self.height = height

        self.block10 = nn.Sequential(
            nn.Linear(self.dim_z, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.block11 = nn.Sequential(
            nn.Linear(1024, 128 * self.width//4 * self.height//4),
            nn.BatchNorm1d(128 * self.width//4 * self.height//4),
            nn.ReLU(True),
        )
        self.block12 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.block13 = nn.Sequential(
            nn.ConvTranspose2d(64, self.dim_c, 2, stride=2),
            nn.Tanh(),
        )

    def decoder(self, x):
        h = self.block10(x)
        h = self.block11(h)
        h = h.view(h.shape[0], 128, self.width//4, self.height//4)
        h = self.block12(h)
        h = self.block13(h)
        return h

    def forward(self, z):
        x = self.decoder(z)
        return 0.5*x + 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--mode', type=str, default='all')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--latent_size', type=int, default=10)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--warm_steps', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.3)
    parser.add_argument('--train_epoch', type=int, default=30)
    parser.add_argument('--ot_batch_size', type=int, default=1000)
    parser.add_argument('--inner_steps', type=int, default=50)
    parser.add_argument('--cuda_ids', nargs='+', help='<Required> Set flag', required=True, default=0)
    parser.add_argument('--step_size', type=float, default=0.3)
    parser.add_argument('--random_size', type=float, default=1.0)
    parser.add_argument('--display', type=bool, default=True)
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--abnormal', type=int, default=9)
    args = parser.parse_args()

    # training parameters
    batch_size = args.batch_size
    lr = args.lr
    train_epoch = args.train_epoch
    latent_size = args.latent_size
    steps = args.steps
    sigma = args.sigma
    checkpoint = args.checkpoint
    test = args.test
    mode = args.mode
    ot_batch_size = args.ot_batch_size
    inner_steps = args.inner_steps
    cuda_ids = list(map(int, args.cuda_ids)) 
    step_size = args.step_size
    random_size = args.random_size
    warm_steps = args.warm_steps
    display = args.display
    abnormal = args.abnormal
    ratio = args.ratio

    print(cuda_ids)
    torch.cuda.set_device(cuda_ids[0])

    save_path = './MNIST/abnormal_' + str(abnormal) + '_' + mode + '_' + str(latent_size) + '_' + str(warm_steps) + '_' + str(steps) + '_' + str(step_size) + '_' + str(inner_steps) + '_' + str(ratio) + '_' + str(random_size) + '_' + str(lr) + '/'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isdir(save_path+'models/'):
        os.mkdir(save_path+'models/')
    if not os.path.isdir(save_path+'mat/'):
        os.mkdir(save_path+'mat/')
    if not os.path.isdir(save_path+'images/'):
        os.mkdir(save_path+'images/')

    logger = logging.getLogger(__name__)  
    logger.setLevel(logging.INFO)
    logging_file = save_path + 'log.log'
    file_handler = logging.FileHandler(logging_file, 'a')
    formatter    = logging.Formatter('%(asctime)s : %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # network
    G = decoder(dim_z=latent_size)
    G = nn.DataParallel(G, device_ids=cuda_ids).cuda()
    if checkpoint>0:
        G.module.load_state_dict(torch.load(save_path+'models/warm_' + str(checkpoint) + '.pth'))
        Z1 = sio.loadmat(save_path+'mat/'+str(checkpoint)+'.mat')
        Z1 = Z1['Z']
        Z1 = torch.from_numpy(Z1).float().cuda()

        Z0 = sio.loadmat(save_path+'mat/Z_tmp_'+str(checkpoint)+'.mat')
        Z0 = Z0['Z']
        Z0 = torch.from_numpy(Z0).float().cuda()

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=[.5, .99])

    # data
    if mode == 'all':
        im = sio.loadmat('./data/mnist.mat')
        im = im['images']
        labels = sio.loadmat('./data/mnist_labels.mat')
        labels = labels['labels']
        labels = np.squeeze(labels)
    else:
        im = sio.loadmat('./data/mnist_0_1_2.mat')
        im = im['images']

    # set the test data
    n_test = 5000
    im_test = im[:n_test]
    im_test = torch.from_numpy(im_test).float()
    l_ = labels[:n_test]
    l_test = l_==abnormal

    # set the training data
    im = im[10000:]
    labels = labels[10000:]
    im = torch.from_numpy(im).float()
    l_abnormal = labels==abnormal
    l_selected = 1 - l_abnormal
    l_selected = [i for i, e in enumerate(l_selected) if e == 1]
    im = im[l_selected]
    n = im.shape[0]

    batch_num = int(n/batch_size)
    if batch_num*batch_size < n:
        batch_num += 1
    OT_thresh = 0.05

    def getloss(pred, x, z, sigma):
        loss = 1/(2*sigma**2) * torch.pow(x - pred, 2).sum() + 1/2 * torch.pow(z, 2).sum()
        loss /= x.size(0)
        return loss

    def auc_test(save_mat=False):
        x = im_test.cuda()
        z_ = torch.randn(n_test, latent_size)
        z_ = Variable(z_, requires_grad=True).cuda()
        for k in range(steps*10):
            out = G(z_)
            loss = getloss(out, x, z_, sigma)
            loss *= step_size**2/2
            delta_z = torch.autograd.grad(loss, z_, retain_graph=True)[0]
            z_.data -= delta_z.data
            z_.data += random_size*step_size*torch.randn((n_test, latent_size)).cuda()

        scores = torch.sum(1/(2*sigma**2) * torch.pow(x - out, 2), (3,2,1)) + torch.sum(1/2 * torch.pow(z_, 2), 1)
        scores = scores.data.cpu().numpy()

        precision, recall, thresholds = precision_recall_curve(l_test, scores)
        auc_ = auc(recall, precision)
        print(auc_, np.sum(l_test))
        if save_mat:
            sio.savemat(save_path+'mat/score.mat', {"score":scores, "label":l_test, "Z_test":z_.cpu().data.numpy()})

        return auc_

    Z0 = torch.randn(n,latent_size).float().cuda()
    Z1 = Z0.clone()
    Z = Z1.clone()
    if not test:
        G.train()     
        for epoch in range(checkpoint, train_epoch):
            for j in range(warm_steps):
                idx = torch.randperm(n)
                G_losses = []
                for i in range(batch_num):       
                    idx0 = i * batch_size
                    idx1 = (i+1) * batch_size
                    if idx1 > n:
                        idx1 = n

                    x = im[idx[idx0:idx1]]
                    mini_batch = x.shape[0]
                    x = x.view(-1, 1, 28, 28).float().cuda()
                         
                    z_ = Z1[idx[idx0:idx1]].cpu()
                    z_ = Variable(z_, requires_grad=True).cuda()
                    for k in range(steps):
                        out = G(z_)
                        loss = getloss(out, x, z_, sigma)
                        loss *= step_size**2/2
                        delta_z = torch.autograd.grad(loss, z_, retain_graph=True)[0]
                        z_.data -= delta_z.data
                        if epoch < train_epoch/1:
                            z_.data += random_size*step_size*torch.randn((mini_batch, latent_size)).cuda()

                    Z1[idx[idx0:idx1]] = z_.data

                    for _ in range(2):
                        x_ = G(z_)
                        G_train_loss = getloss(x_, x, z_, sigma)
                        loss1 = torch.pow(x - x_, 2).sum() / x.size(0)
                        G_optimizer.zero_grad()
                        G_train_loss.backward()
                        G_optimizer.step()
                        G_losses.append(loss1.data.item())

                G_losses = torch.FloatTensor(G_losses)
                if display:
                    print(epoch, j, torch.mean(G_losses))
                logger.info('epoch: %d, warm step: %d, loss: %5f', epoch, j, torch.mean(G_losses))
            
            auc_test()

            z = torch.randn(64, latent_size).cuda()
            x_ = G(z)
            utils.save_image(x_[:64], save_path+'images/warm_'+str(epoch)+'_gen.png', nrow=8)
            torch.save(G.module.state_dict(), save_path+'models/warm_'+str(epoch)+'.pth')
            sio.savemat(save_path+'mat/Z_tmp_'+str(epoch)+'.mat', {"Z": Z1.cpu().data.numpy()})
                    
            if epoch == 0:
                h, E, area_diff, hyper_num = OT_solver(Z1, Z0, ot_batch_size, OT_thresh=OT_thresh, loops=20000, display=True)
                index, source_idx = correspondence_reconstruction(Z1, Z0, h)
            if epoch >=1:
                h, E, area_diff, hyper_num = OT_solver(Z1, Z0, ot_batch_size, heights=h, loops=20000, OT_thresh=OT_thresh, display=True)
                index, source_idx = correspondence_reconstruction(Z1, Z0, h)

            
            Z = Z0[index[source_idx]]
            OT_cost = torch.pow(Z1[source_idx]-Z,2).sum()/(Z1.size(0))
            OT_cost = OT_cost.data.cpu()
            Z = Z * ratio + Z1[source_idx]*(1-ratio)
            Z1 = Z0[index]*ratio + Z1*(1-ratio)

            _im = im[source_idx]

            for j in range(inner_steps):
                idx = torch.randperm(_im.shape[0])
                G_losses = []
                for i in range(batch_num):
                    idx0 = i * batch_size
                    idx1 = (i+1) * batch_size
                    if idx1 > _im.shape[0]:
                        break
                    if idx1 > n:
                        idx1 = n

                    x = _im[idx[idx0:idx1]]
                    mini_batch = x.shape[0]
                    x = x.view(-1, 1, 28, 28).float().cuda()
                    z_ = Z[idx[idx0:idx1]]

                    x_ = G(z_)
                    G_train_loss = torch.pow(x - x_, 2).sum() / x.size(0)
                    G_losses.append(G_train_loss.data.item())
                    
                    G_optimizer.zero_grad()
                    G_train_loss.backward()
                    G_optimizer.step()

            G_losses = torch.FloatTensor(G_losses)
            torch.save(G.state_dict(), save_path+'models/'+str(epoch)+'_'+str(j)+'.pth')

            auc_test()
            
            # save both the reconstructed and generated images
            utils.save_image(x_[:64], save_path+'images/'+str(epoch)+'.png', nrow=8)
            z = torch.randn(64, latent_size).cuda()
            x_ = G(z)
            utils.save_image(x_[:64], save_path+'images/'+str(epoch)+'_gen.png', nrow=8)

            # save the model G and the reordered Gaussian samples
            if epoch % 2 == 0:
                torch.save(G.state_dict(), save_path+'models/OT_'+str(epoch)+'.pth')
                sio.savemat(save_path+'mat/'+str(epoch)+'.mat', {"Z": Z1.cpu().data.numpy()})
            if display:
                print('[%d/%d]: loss_d: %.3f OT_loss: %.5f, area_diff: %.5f hyper_num: %.1f' % (
                    (epoch + 1), train_epoch, torch.mean(G_losses), OT_cost, area_diff, hyper_num))

            logger.info('[%d/%d]: loss_d: %.3f, OT_loss: %.5f, area_diff: %.5f hyper_num: %.1f', 
                epoch + 1, train_epoch, torch.mean(G_losses), OT_cost, area_diff, hyper_num)



        
