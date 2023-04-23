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


#### (2) Testing


1. To generate images using pretrained models, please first download the pretrained checkpoints from "[this link](https://drive.google.com/drive/folders/14OtnJpIhiiH9UT3kCSLPllDyrV3iop7j?usp=share_link)". The folder contains checkpoints with different datasets. 

The checkpoints should be downloaded to the ckpt folder (e.g., you should have 'ckpt/cifar10.pth.tar' for CoopFlow cifar10 setting).



train_svhn: contain the first version of flow that outputs the latent vector

update the 256 structure:

#### code: 
```bash
python train_celeba256.py
```

#### dataset:
The download link for CelebA-HQ seems to broke. Thus, I finally use the CelebA-Mask-HQ "[dataset](https://github.com/switchablenorms/CelebAMask-HQ)", which is a similar large resolution dataset.
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

The last 2 lines remove some used information. This enables the data to be loaded in correctly.
After download the data, make sure in line 301 and 302, you data path point to the CelebAMask-HQ folder.
