
from train import train, test, copy_source, to_named_dict, get_exp_id
import argparse 
import pygrid
import os

def parse_args_cifar10():

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train", help='training or test mode')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--abnormal', type=int, default=-1, help='training or test mode')
    parser.add_argument('--load_checkpoint', type=str, default="", help='load checkpoint')
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--device', type=int, default=0, help='training or test mode')
    parser.add_argument('--output_dir', type=str, default="default", help='training or test mode')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['svhn', 'celeba', 'celeba_crop', 'mnist', 'mnist_ad', 'cifar10'])
    parser.add_argument('--incomplete_train', type=int, default=0, help='training or test mode')
    parser.add_argument('--data_size', type=int, default=1000000)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ngf',type=int,  default=128, help='feature dimensions of generator')

    parser.add_argument('--g_llhd_sigma', type=float, default=0.3, help='prior of factor analysis')
    parser.add_argument('--g_activation', type=str, default='lrelu')
    parser.add_argument('--g_activation_leak', type=float, default=0.2)
    parser.add_argument('--g_l_steps', type=int, default=40, help='number of langevin steps')
    parser.add_argument('--g_l_step_size', type=float, default=0.1, help='stepsize of langevin') # 0.1
    parser.add_argument('--g_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--g_batchnorm', default=False, type=bool, help='batch norm')

    parser.add_argument('--f_n_levels', default=1, type=int, help='')
    parser.add_argument('--f_depth', default=5, type=int, help='') # 10
    parser.add_argument('--f_flow_permutation', default=2, type=int, help='')
    parser.add_argument('--f_width', default=64, type=int, help='')
    parser.add_argument('--f_flow_coupling', default=1, type=int, help='')

    parser.add_argument('--g_lr', default=0.00038, type=float) # 0.0004
    parser.add_argument('--f_lr', default=0.00038, type=float) # 0.0004

    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--f_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')

    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--f_max_norm', type=float, default=100, help='max norm allowed')

    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')
    parser.add_argument('--f_decay', default=0, help='weight decay for flow')

    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for gen')
    parser.add_argument('--f_gamma', type=float, default=0.998, help='lr decay for flow')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--f_beta1', default=0.5, type=float)
    parser.add_argument('--f_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=20, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')

    parser.add_argument('--n_ckpt', type=int, default=1, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')    #
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')
    parser.add_argument('--n_fid_samples', type=int, default=50000)


    return parser.parse_args()

def parse_args_svhn():

    parser.add_argument('--mode', type=str, default="train", help='training or test mode')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--abnormal', type=int, default=-1, help='training or test mode')
    parser.add_argument('--load_checkpoint', type=str, default="", help='load checkpoint')
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--device', type=int, default=0, help='training or test mode')
    parser.add_argument('--output_dir', type=str, default="default", help='training or test mode')
    parser.add_argument('--dataset', type=str, default='svhn', choices=['svhn', 'celeba', 'celeba_crop', 'mnist', 'mnist_ad'])
    parser.add_argument('--incomplete_train', type=int, default=0, help='training or test mode')
    parser.add_argument('--data_size', type=int, default=1000000)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--ngf',type=int,  default=64, help='feature dimensions of generator')

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

    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')
    parser.add_argument('--f_decay', default=0, help='weight decay for flow')

    parser.add_argument('--g_gamma', type=float, default=0.998, help='lr decay for gen')
    parser.add_argument('--f_gamma', type=float, default=0.998, help='lr decay for flow')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--f_beta1', default=0.5, type=float)
    parser.add_argument('--f_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=20, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')

    parser.add_argument('--n_ckpt', type=int, default=1, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')    #
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')
    parser.add_argument('--n_fid_samples', type=int, default=50000)

def parse_args_celeba():

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--abnormal', type=int, default=-1, help='training or test mode')
    parser.add_argument('--mode', type=str, default="train", help='training or test mode')
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument('--device', type=int, default=0, help='training or test mode')
    parser.add_argument('--output_dir', type=str, default="default", help='training or test mode')
    parser.add_argument('--incomplete_train', type=int, default=0, help='training or test mode')
    parser.add_argument('--data_size', type=int, default=1000000)
    parser.add_argument('--load_checkpoint', type=str, default="", help='load checkpoint')
    parser.add_argument('--dataset', type=str, default='celeba_crop', choices=['svhn', 'celeba', 'celeba_crop'])
    parser.add_argument('--img_size', default=64, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--nc', default=3)
    parser.add_argument('--ngf', default=128, help='feature dimensions of generator')

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

    parser.add_argument('--g_lr', default=0.0003, type=float) # 0.0004
    parser.add_argument('--f_lr', default=0.0003, type=float) # 0.0004

    parser.add_argument('--g_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--f_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')

    parser.add_argument('--g_max_norm', type=float, default=100, help='max norm allowed')
    parser.add_argument('--f_max_norm', type=float, default=100, help='max norm allowed')

    parser.add_argument('--g_decay',  default=0, help='weight decay for gen')
    parser.add_argument('--f_decay', default=0, help='weight decay for flow')

    parser.add_argument('--g_gamma', default=0.998, help='lr decay for gen')
    parser.add_argument('--f_gamma', default=0.998, help='lr decay for flow')

    parser.add_argument('--g_beta1', default=0.5, type=float)
    parser.add_argument('--g_beta2', default=0.999, type=float)

    parser.add_argument('--f_beta1', default=0.5, type=float)
    parser.add_argument('--f_beta2', default=0.999, type=float)

    parser.add_argument('--n_epochs', type=int, default=201, help='number of epochs to train for')
    parser.add_argument('--n_printout', type=int, default=20, help='printout each n iterations')
    parser.add_argument('--n_plot', type=int, default=1, help='plot each n epochs')

    parser.add_argument('--n_ckpt', type=int, default=1, help='save ckpt each n epochs')
    parser.add_argument('--n_metrics', type=int, default=1, help='fid each n epochs')    #
    parser.add_argument('--n_stats', type=int, default=1, help='stats each n epochs')
    parser.add_argument('--n_fid_samples', type=int, default=10000)


    return parser.parse_args()

def main():

    opt = {'job_id': int(0), 'status': 'open'}
    args = parse_args()
    args = pygrid.overwrite_opt(args, opt)
    args = to_named_dict(args)
    output_dir = pygrid.get_output_dir(get_exp_id(__file__), fs_prefix='./') if args.output_dir == "default" else ("output/train_celeba/" + args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + '/samples')
        os.makedirs(output_dir + '/ckpt')
    copy_source(__file__, output_dir)
    path_check_point = None if args.load_checkpoint == "" else args.load_checkpoint

    if args.mode == "train":
        # training mode
        copy_source(__file__, output_dir)
        train(args, output_dir, path_check_point)
    elif args.mode == "test":
        # testing mode
        test(args, output_dir, path_check_point)

if __name__ == '__main__':
    main()