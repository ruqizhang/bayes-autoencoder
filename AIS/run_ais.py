"""
author: wesley maddox
date: 10/1/18
bastardized version of bdmc.py in the original git repo
"""

import numpy as np
import itertools
import time

import torch
from torch.autograd import Variable
from torch.autograd import grad as torchgrad
from ais import ais_trajectory
from simulate import simulate_data
#from vae import VAE
from utils import sigmoidal_schedule

import sys
sys.path.append('..')
import unsup.models as models
import unsup.data as unsup_data
#sys.path.append('/nfs01/wm326/bvae/')
#import vae_images.mnist_unsup.models as image_models

#from hparams import get_default_hparams
from torchvision import datasets, transforms
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='AIS with bayesian auto-encoder')
parser.add_argument('--file', type=str, default ='', help = 'file to run test loading on')
parser.add_argument('--dataset', type=str, default='MNIST')
parser.add_argument('--model', type=str, default='BAE', help = 'class of model to load')
parser.add_argument('--zdim', type=int, default = 20, metavar = 'S',
                    help='latent + noise dimension to use in model')
parser.add_argument('--num-steps', type=int, default = 500, help = 'number of steps to run AIS for')
parser.add_argument('--num-samples', type=int, default = 16, help='number of chains to run AIS over')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', help='location of mnist dataset')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--seed', type=int, default=1, help = 'random seed to use')
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
torch.manual_seed(1)


def forward_ais(model, loader, forward_schedule=np.linspace(0., 1., 500), n_sample=100):
    """Bidirectional Monte Carlo. Integrate forward and backward AIS.
    The backward schedule is the reverse of the forward.

    Args:
        model (vae.VAE): VAE model
        loader (iterator): iterator to loop over pairs of Variables; the first 
            entry being `x`, the second being `z` sampled from the true 
            posterior `p(z|x)`
        forward_schedule (list or numpy.ndarray): forward temperature schedule;
            backward schedule is used as its reverse
    Returns:
        Two lists for forward and backward bounds on batchs of data
    """

    # iterator is exhaustable in py3, so need duplicate
    load, _ = itertools.tee(loader, 2)

    # forward chain
    forward_logws = ais_trajectory(model, load, mode='forward', schedule=forward_schedule, n_sample=n_sample)

    lower_bounds = []

    for forward in forward_logws:
        lower_bounds.append(forward.mean())

    lower_bounds = np.mean(lower_bounds)


    return forward_logws

def construct_model_and_loader():
    ###############################################################################
    #define the model
    ###############################################################################
    print('Using model %s' % args.model)
    model_cfg = getattr(models, args.model)

    ###############################################################################
    # Load data
    ###############################################################################
    loaders, ntokens = unsup_data.loaders(args.dataset, args.data_path, args.batch_size, args.bptt, 
                            model_cfg.transform_train, model_cfg.transform_test,
                            use_validation=False, use_cuda=True)

    ###############################################################################
    # Build the model
    ###############################################################################
    if args.num_samples > 1:
        model_batch_size = args.num_samples * args.batch_size
    else:
        model_batch_size = args.batch_size
    print('Preparing model')
    print(*model_cfg.args)
    print('using ', args.zdim, ' latent space')
    model = model_cfg.base(*model_cfg.args, 
                        noise_dim=args.zdim, zdim=args.zdim, 
                        ntoken=ntokens, ninp=200, nhidden=args.nhid, 
                        nlayers=args.nlayers, bsz=model_batch_size, 
                        dropout=args.dropout, tie_weights=args.tied,
                        **model_cfg.kwargs)
    model.cuda()

    if args.dataset == 'MNIST' and args.batch_size == 10000:
        # store data + label on cuda for speed
        for (data, label) in loaders['test']:
            test_tensor_list = (data.cuda(async=False).view(-1, 784), label.cuda(async=False))
        class myiterator:
            def __iter__(self):
                return iter([test_tensor_list])

        loaders['test'] = myiterator()

    return model, loaders['test']
       
def main(f=args.file):
    torch.manual_seed(args.seed)

    model, loader = construct_model_and_loader()
    
    print('Loading provided model')
    # there may be discrepancies in how the model was saved
    try:
        model_state_dict = torch.load(f, map_location=lambda storage, loc: storage)['state_dict']
    except:
        model_state_dict = torch.load(f, map_location=lambda storage, loc: storage)

    model.load_state_dict(model_state_dict)

    #model.eval()

    # run num_steps of AIS in batched mode with num_samples chains    
    # sigmoidal schedule is typically used
    logws = forward_ais(model, loader, forward_schedule=sigmoidal_schedule(args.num_steps), n_sample=args.num_samples)
    batch_logw = [logw.mean().cpu().data.numpy() for logw in logws]
    logw_full = np.mean(batch_logw)
    np.savez('text_results/'+args.model+'_seed_'+str(args.seed)+'.npz', logws=logw_full)

if __name__ == '__main__':
    main()