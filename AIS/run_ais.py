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
from vae import VAE
from utils import sigmoidal_schedule

import sys
sys.path.append('..')
from unsup_text.models import *
#from bae import BAE
sys.path.append('/nfs01/wm326/bvae/vae_images')
from mnist_unsup.models import *

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
to_bernoulli = lambda x: transforms.ToTensor()(x).bernoulli()

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
    #use a variant of test loader so we don't have to re-generate the batch stuff
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('/scratch/datasets', train=False, transform=to_bernoulli, download=True),
        batch_size=10000, shuffle=True)

    for (data, label) in test_loader:
        test_tensor_list = (data.cuda(async=False).view(-1, 784), label.cuda(async=False))
    class myiterator:
        def __iter__(self):
            return iter([test_tensor_list])

    loader = myiterator()

if args.dataset == 'ptb':
    import unsup_text.data as data
    corpus = data.Corpus('../datasets/ptb') #change this line if necessary

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

    loader = batchify(corpus.test, eval_batch_size)

hps = {'dim':784,'zdim':args.zdim,'hidden':500,'noise_dim':args.zdim}
#hps = {'z_dim':args.zdim, 'noise_dim':args.zdim, 'dim':784}

def main(f=args.file):
    #hps = get_default_hparams()
    #model = VAE(hps)
    if args.model == 'VAE':
        model = VAE(**hps)
    if args.model == 'BAE':
        model = BayesAE(**hps)

    # load model
    model.cuda()
    # there may be discrepancies in how the model was saved
    try:
        model_state_dict = torch.load(f, map_location=lambda storage, loc: storage)['state_dict']
    except:
        model_state_dict = torch.load(f, map_location=lambda storage, loc: storage)

    model.load_state_dict(model_state_dict)

    model.eval()

    # run num_steps of AIS in batched mode with num_samples chains    
    forward_ais(model, loader, forward_schedule=sigmoidal_schedule(args.num_steps), n_sample=args.num_samples)


if __name__ == '__main__':
    main()