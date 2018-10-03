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
import unsup_text.models as text_models
import unsup_images.models as image_models
#sys.path.append('/nfs01/wm326/bvae/')
#import vae_images.mnist_unsup.models as image_models

#from hparams import get_default_hparams
from torchvision import datasets, transforms

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
    load, load_ = itertools.tee(loader, 2)

    # forward chain
    forward_logws = ais_trajectory(model, load, mode='forward', schedule=forward_schedule, n_sample=n_sample)

    lower_bounds = []

    for forward in forward_logws:
        lower_bounds.append(forward.mean())

    lower_bounds = np.mean(lower_bounds)


    return forward_logws

if args.dataset == 'MNIST':
    to_bernoulli = lambda x: transforms.ToTensor()(x).bernoulli()

    #use a variant of test loader so we don't have to re-generate the batch stuff
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_path, train=False, transform=to_bernoulli, download=True),
        batch_size=10000, shuffle=True)

    # store data + label on cuda for speed
    for (data, label) in test_loader:
        test_tensor_list = (data.cuda(async=False).view(-1, 784), label.cuda(async=False))
    class myiterator:
        def __iter__(self):
            return iter([test_tensor_list])

    loader = myiterator()

    print('Using model %s' % args.model)
    model_cfg = getattr(image_models, args.model)

if args.dataset == 'ptb':
    import unsup_text.data as text_data
    corpus = text_data.Corpus(args.data_path)

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if args.cuda:
            data = data.cuda(async=True)
        return data

    #eval batch size is hardcoded to 64 to match old bae_ais.py script
    loader = batchify(corpus.test, 64) 
    
    print('Using model %s' % args.model)
    model_cfg = getattr(text_models, args.model)

def main(f=args.file):
    print('Preparing model')
    print(*model_cfg.args)
    print('using ', args.zdim, ' latent space')
    model = model_cfg.base(*model_cfg.args, zdim=args.zdim, **model_cfg.kwargs)
    #model.to(args.device)
    model.cuda()
    #model.device = args.device
    
    print('Loading provided model')
    # there may be discrepancies in how the model was saved
    try:
        model_state_dict = torch.load(f, map_location=lambda storage, loc: storage)['state_dict']
    except:
        model_state_dict = torch.load(f, map_location=lambda storage, loc: storage)

    model.load_state_dict(model_state_dict)

    model.eval()

    # run num_steps of AIS in batched mode with num_samples chains    
    # sigmoidal schedule is typically used
    forward_ais(model, loader, forward_schedule=sigmoidal_schedule(args.num_steps), n_sample=args.num_samples)


if __name__ == '__main__':
    main()