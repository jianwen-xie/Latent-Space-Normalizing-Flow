# Latent Space Flow-Based Model

This repository contains a pytorch implementation for AAAI 2023 paper "[A Tale of Two Latent Flows: Learning Latent Space Normalizing Flow with Short-run Langevin Flow for Approximate Inference](https://arxiv.org/pdf/2301.09300.pdf)"

## Set Up Environment
We have provided the environment.yml file for setting up the environment. The environment can be set up with one command using conda

```bash
conda env create -f environment.yml
conda activate fpp
```

## Reference
    @article{DG,
        author = {Jianwen Xie, Yaxuan Zhu, Yifei Xu, Dingcheng Li, Ping Li},
        title = {A Tale of Two Latent Flows: Learning Latent Space Normalizing Flow with Short-run Langevin Flow for Approximate Inference},
        journal={The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI)},
        year = {2023}
    }
    
    
## Usage

### (1) Image Generation

#### (i) Training

(a) SVHN dataset

    $ python train.py --dataset svhn --g_l_steps 20 --img_size 32 --nz 100 --ngf 64 --g_lr 0.0004  --f_lr 0.0004

    
(b) Cifar-10 dataset

    $ python train.py --dataset cifar10 --g_l_steps 40 --img_size 32 --nz 128 --ngf 128 --g_lr 0.00038 --f_lr 0.00038
    
(c) CelebA dataset

    $ python train.py --dataset celeba_crop --g_l_steps 20 --img_size 64 --nz 100 --ngf 128 --g_lr 0.0003 --f_lr 0.0003 
    

#### (ii) Testing


To generate images using pretrained models, please first download the pretrained checkpoints from "[this link](https://drive.google.com/drive/folders/14OtnJpIhiiH9UT3kCSLPllDyrV3iop7j?usp=share_link)". The folder contains checkpoints with different datasets. 

The checkpoints should be downloaded to the ./ckpt folder, e.g., you should have './ckpt/ckpt_000115.pth' for the experiment using SVHN dataset. The following commands synthesize 50,000 new images and evaluate the FIDs. If you want to test the reconstruction ability. Please add '--testing_reconstruct' into the commands and it will reconstruct testing sets and report reconstruction errors.     

(a) SVHN dataset

    $ python train.py --dataset svhn --test_mode --g_l_steps 400 --img_size 32 --nz 100 --ngf 64 --g_lr 0.0004  --f_lr 0.0004 --path_check_point ./ckpt/ckpt_000115.pth --n_fid_samples 50000

    
(b) Cifar-10 dataset

    $ python train.py --dataset cifar10 --test_mode --g_l_steps 800 --img_size 32 --nz 128 --ngf 128 --g_lr 0.00038 --f_lr 0.00038 --path_check_point ./ckpt/ckpt_000093.pth --n_fid_samples 50000
    
(c) CelebA dataset

    $ python train.py --dataset celeba_crop --test_mode --g_l_steps 400 --img_size 64 --nz 100 --ngf 128 --g_lr 0.0003 --f_lr 0.0003 --path_check_point ./ckpt/ckpt_000071.pth --n_fid_samples 50000
    

#### (iii) 256 x 256 Image Generation 

CelebA-HQ dataset

    $ python train.py --dataset celeba_hq256 --g_l_steps 20 --img_size 256 --nz 100 --ngf 128 --g_llhd_sigma 1.0 --f_width 128 --g_lr 0.0003 --f_lr 0.0003 



#### dataset:
The download link for CelebA-HQ seems to broke. Thus, we finally use the CelebA-Mask-HQ "[dataset](https://github.com/switchablenorms/CelebAMask-HQ)", which is a similar large resolution dataset.

To download the data:

```bash
mkdir ./data
cd data
gdown https://drive.google.com/uc?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv
unzip CelebAMask-HQ.zip 
cd CelebAMask-HQ
rm *.txt
rm -r -f CelebAMask-HQ-mask-anno/
```

After download the data, make sure that the data path points to the CelebAMask-HQ folder.
