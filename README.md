train_svhn: contain the first version of flow that outputs the latent vector

update the 256 structure:

###code: 
```bash
python train_celeba256.py
```

###dataset:
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
