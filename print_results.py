
import math 
import numpy as np
import matplotlib
import sys
sys.path.append('.')
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
font_size = 15
matplotlib.rc('xtick', labelsize=font_size) 
matplotlib.rc('ytick', labelsize=font_size) 
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import utils.cost_dynamic as cost_dynamic
from utils.util_torch import *
import model.network as NNModel
from scipy.stats import norm
import torch 

def print_graph():

    x = [3,     5,      16,     32,     64,     128,    256,    512,    1024,   2048,   
        4096,  10000, 16000, 32000]
    y = [8.260, 6.548,  0.711,  0.786,  0.700,  0.654,  0.647,  0.645,  0.649,  0.629,  
        0.626, 0.669, 0.668, 0.637]
    plt.plot(x, y)
    plt.plot(x, [0.872] * 14)
    # plt.legend(loc=1, prop={'size': font_size})
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of training data", fontsize=font_size)
    plt.ylabel("RMSE", fontsize=font_size)
    # plt.ylim([np.min(np.array(rmse_list)) - 0.05, rmse_list[i][0] + 0.1])
    plt.savefig("output/robust_training.png")
    plt.savefig("output/robust_training.eps")
    plt.clf()

def demo_control_output(category="nips0610_isee"):

    with open("output/%s/config.pkl" % category, "rb") as conf:
        opt = pickle.load(conf)

    test_data = np.load('data/package_12.npy')[:, 1:opt.traj_size+1]
    print(test_data.shape)
    opt.multiple_factor = multiple_factor = [92, 89]
    test_data[..., 4] *= multiple_factor[0]
    test_data[..., 6] *= multiple_factor[0]
    test_data[..., 5] *= multiple_factor[1]
    test_data[..., 7] *= multiple_factor[1]
    num_data = 1
    test_status, test_extend = torch.Tensor(test_data[:num_data, :, :8]), torch.Tensor(test_data[:num_data, :, 8:])
    des_model, _ = torch.load(os.path.join('./output', category, "model_200.pt"))
    des_model = des_model.cpu()
    des_model.weight = des_model.weight.cpu()
    dynamic =  cost_dynamic.CarDynamicDefault(opt.traj_size, opt.multiple_factor).car_dynamic_pt

    low, high, interval = -1, 1, 0.02
    x = torch.autograd.Variable(torch.Tensor(np.array(np.random.uniform(low, high, size=(num_data)))), requires_grad=True)
    x_print = np.arange(low, high, interval)
    print(x_print.shape)
    y_op = lambda inp: des_model(dynamic(test_status.repeat(inp.shape[0],1,1), 
    torch.stack([inp.repeat(opt.traj_size, 1), torch.zeros((opt.traj_size, inp.shape[0]))], axis=2).permute([1,0,2])), test_extend.repeat(inp.shape[0],1,1))

    gt = torch.autograd.Variable(torch.Tensor(np.array(np.random.uniform(low, high, size=(num_data)))))
    y_print_list = [y_op(torch.Tensor(x_print)).data.numpy()]
    x_his_list = []
    y_his_list = []
    gty_list = []

    dt = 1e-1
    p_op = torch.optim.SGD(des_model.parameters(), lr=1e-3)
    
    for epoch in range(100):

        x = torch.autograd.Variable(torch.Tensor(np.array(np.random.uniform(low, high, size=(num_data)))), requires_grad=True)
        x_his = []
        for s_step in range(20):
            x_loss = torch.sum(y_op(x))
            grad = torch.autograd.grad(x_loss, [x], retain_graph=True)[0]
            noise = torch.randn(1)
            x = x - (0.5 * dt * dt * grad + dt * noise)
            x_his.append(float(x.data.numpy()))

        # print(x_his)
        y_print = y_op(torch.Tensor(x_print)).data.numpy()
        y_print_list.append(y_print)
        x_his_list.append(x_his)
        y_his_list.append(y_op(torch.Tensor(x_his)).data.numpy())
        gty_list.append(y_op(gt).data.numpy())

        loss = torch.sum(y_op(gt)) - torch.sum(y_op(x))
        p_op.zero_grad()
        loss.backward()
        p_op.step()
        # plt.clf()
        # plt.plot(x_print, y_print)
        # plt.plot(x_his, y_op(torch.Tensor(x_his)).data.numpy(), 'r.')
        # plt.plot(gt.data.numpy(), y_op(gt).data.numpy(), 'g.')
        # # plt.axis("off")
        # plt.savefig(output_name[:-4] + "_%d.png" % epoch)
    output_gif(x_print, y_print_list, x_his_list, y_his_list, gt.data.numpy(), gty_list, "output/demo_con/control.png")



def output_gif(x_print, y_print_list, x_his_list, y_his_list, gt, gt_list, output_name):
    rc('animation', html='html5')
    fig, ax = plt.subplots()
    ax.set_xlim((-1, 1))
    y_print_list = np.array(y_print_list)
    ax.set_ylim((np.min(y_print_list), np.max(y_print_list)))
    lines = []
    line, = ax.plot([], [])
    lines.append(line)
    line, = ax.plot([], [], 'r.', lw=5)
    lines.append(line)
    line, = ax.plot([], [], 'g.', lw=5)
    lines.append(line)
    def init():
        for line in lines:
            line.set_data([], [])
        return (lines,)
    def animate(ii):
        lines[0].set_data(x_print, y_print_list[ii+1])
        lines[1].set_data(x_his_list[ii], y_his_list[ii])
        lines[2].set_data(gt, gt_list[ii])
        ax.set_title("Frame %d" % ii)
        return (lines,)
    print('Saving %s' % output_name)
    anim = animation.FuncAnimation(fig, animate, frames=len(y_his_list), interval=10, blit=False)
    anim.save("%s.gif" % (output_name[:-4]), writer=animation.PillowWriter(fps=3))

def output_multi_normal(output_name, list_mean, list_var, list_scale, low=0, high=1, interval=1e-4):

    print("Printing %s" % output_name)
    x = np.arange(low, high, interval)
    y = np.array([norm.pdf(x, list_mean[i], list_var[i]) * list_scale[i] for i in range(len(list_mean))])
    ny = np.sum(y, 0)
    plt.clf()
    plt.plot(x, np.max(y) - y)
    plt.axis("off")
    plt.savefig(output_name)

    lr = 0.1 

def plot(out_name, *args, **kwargs):

    plt.clf()
    plt.plot(*args, **kwargs)
    if "axis" in kwargs.keys():
        plt.axis(kwargs["axis"])
    else:
        plt.axis("off")
    plt.savefig(out_name)

def sample_langevin(output_name, list_mean, list_var, list_scale, low=0, high=1, interval=1e-4):

    x = torch.autograd.Variable(torch.Tensor(np.array(np.random.uniform(low, high))), requires_grad=True)
    x_print = np.arange(low, high, interval)
    gt = torch.autograd.Variable(torch.Tensor(np.array(np.random.uniform(low, high))))
    parameter = torch.autograd.Variable(torch.Tensor(np.stack([list_mean, list_var, list_scale])), requires_grad=True)
    def norm_prob(value, mean, std):
        return torch.exp(-((value - mean) ** 2) / (2 * torch.exp(std)**2) - std - math.log(math.sqrt(2 * math.pi)))
    y_op = lambda inp: - torch.sum(torch.stack([norm_prob(inp, parameter[0, i], parameter[1, i]) * parameter[2, i] for i in range(len(list_mean))]), dim=0)
    
    y_print_list = [y_op(torch.Tensor(x_print)).data.numpy()]
    x_his_list = []
    y_his_list = []
    gty_list = []
    # plot(output_name, x_print, y_print_list[0])

    dt = 1e-2
    p_op = torch.optim.SGD([parameter], lr=1e-4)
    
    for epoch in range(20):

        x = torch.autograd.Variable(torch.Tensor(np.array(np.random.uniform(low, high))), requires_grad=True)
        x_his = []
        for s_step in range(20):
            x_loss = y_op(x)
            grad = torch.autograd.grad(x_loss, [x], retain_graph=True)[0]
            noise = torch.randn(1)
            x = x - (0.5 * dt * dt * grad + dt * noise)
            x_his.append(float(x.data.numpy()))

        # print(x_his)
        y_print = y_op(torch.Tensor(x_print)).data.numpy()
        y_print_list.append(y_print)
        x_his_list.append(x_his)
        y_his_list.append(y_op(torch.Tensor(x_his)).data.numpy())
        gty_list.append(y_op(gt).data.numpy())

        loss = y_op(gt) - y_op(x)
        p_op.zero_grad()
        loss.backward()
        p_op.step()
        # plt.clf()
        # plt.plot(x_print, y_print)
        # plt.plot(x_his, y_op(torch.Tensor(x_his)).data.numpy(), 'r.')
        # plt.plot(gt.data.numpy(), y_op(gt).data.numpy(), 'g.')
        # # plt.axis("off")
        # plt.savefig(output_name[:-4] + "_%d.png" % epoch)
    output_gif(x_print, y_print_list, x_his_list, y_his_list, gt.data.numpy(), gty_list, output_name)

def demo_output():

    n_out = 100
    for seed in range(n_out):
        np.random.seed(seed)
        torch.manual_seed(seed)
        n = 20
        sample_langevin("lan/normal_%d.png" % seed, np.random.uniform(size=(n)), np.random.uniform(-4,0, size=(n)), np.random.uniform(0, 0.01, size=(n)))

# print_graph()
demo_control_output()