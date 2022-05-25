import pytorch_fid_wrapper as pfw
import torch
import os
import numpy as np

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

from tqdm import tqdm
from pytorch_fid_wrapper.inception import InceptionV3
from pytorch_fid_wrapper.fid_score import get_activations

def new_comparison_method(path=None):

    test_data = torch.from_numpy(np.load("/home/fei960922/Documents/UCLA_reference/reference_CV/Flow-Based-Prior-Model/output/incomplete_truth.npy"))
    res_data = torch.from_numpy(np.load("/home/fei960922/Documents/UCLA_reference/reference_CV/Flow-Based-Prior-Model/output/incomplete_save_all_truth_0.npy"))
    print("Total test data: %d; total results: %d" % (test_data.shape[0], res_data.shape[1]))

    device = "cuda:1"
    DIM = 2048
    batch_size = 100
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[DIM]]).to(device)
    total_fid = 0

    # Calculate real activation distributions
    print("Calculating activation distributions...")
    act_real = []
    for i in range(100):
        act = get_activations(test_data[i*batch_size:(i+1)*batch_size], model, batch_size=batch_size, dims=DIM, device=device)
        act_real.append(act)
        fake_data = res_data[:, i*batch_size:(i+1)*batch_size].reshape(res_data.shape[0]*batch_size, 3, res_data.shape[3], res_data.shape[4])
        act_fake = get_activations(fake_data, model, batch_size=batch_size, dims=DIM, device=device).reshape(res_data.shape[0], batch_size, DIM)
        fid_mse = []
        for j in range(10):
            fid_mse.append(np.mean((act - act_fake[j])**2))
        # fid_mse = np.concatenate(fid_mse)
        fid_mse = np.mean(fid_mse, axis=0)
        total_fid += fid_mse
        print("# %d / 100: Current average FID: %.4f, this: %.4f" % (i, fid_mse, total_fid / (i+1)))
    print("%.4f" % (total_fid / 100))

if __name__ == '__main__':
    new_comparison_method()