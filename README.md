# Flow based prior model

## Train

- Now support "abnormal" args to delete a specific anormaly category while training. Used with mnist and anormaly detection only.

train_svhn: contain the first version of flow that outputs the latent vector

## Anomaly detection

- Train with "abnormal" label. e.g. set it with 1
  - `python train_mnist.py --abnormal 1`
- Then run the testing with the following:
  - `python train_mnist.py --model test --abnormal 1 --load_checkpoint output/{PATH_TO_PTH}`

Please make sure two query use the same parameters.




## Inpainting


```python

python train.py --mode test --dataset celeba_crop --img_size 64 --ngf 128 --load_checkpoint /home/fei960922/Documents/UCLA_reference/reference_CV/Flow-Based-Prior-Model/output/celeba_vae_flow.pth --g_l_steps 400 --g_l_step_size 0.002
```
