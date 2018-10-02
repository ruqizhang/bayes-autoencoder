import numpy as np
import itertools
import time

import torch
from torch.autograd import Variable
from torch.autograd import grad as torchgrad
from ais import ais_trajectory
from simulate import simulate_data
from vae import VAE

import sys
sys.path.append('..')
from bae_model import BAE

#from hparams import get_default_hparams
from torchvision import datasets, transforms

import argparse

parser = argparse.ArgumentParser(description='AIS with bayesian auto-encoder')
parser.add_argument('--file', type=str, default ='', help = 'file to run test loading on')
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
        forward_schedul/home/rz297/vae/results/bn_model.pte (list or numpy.ndarray): forward temperature schedule;
            backward schedule is used as its reverse
    Returns:
        Two lists for forward and backward bounds on batchs of data
    """

    # iterator is exhaustable in py3, so need duplicate
    load, load_ = itertools.tee(loader, 2)

    # forward chain
    forward_logws = ais_trajectory(model, load, mode='forward', schedule=forward_schedule, n_sample=n_sample)

    # backward chain
    #backward_schedule = np.flip(forward_schedule, axis=0)
    #backward_logws = ais_trajectory(model, load_, mode='backward', schedule=backward_schedule, n_sample=n_sample)

    #upper_bounds = []
    lower_bounds = []

    for forward in forward_logws:
        lower_bounds.append(forward.mean())
        #upper_bounds.append(backward.mean())

    #upper_bounds = np.mean(upper_bounds)
    lower_bounds = np.mean(lower_bounds)

    #print ('Average bounds on simulated data: lower %.4f, upper %.4f' % (lower_bounds, upper_bounds))

    return forward_logws#, backward_logws

#use a variant of test loader so we don't have to re-generate the batch stuff
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/scratch/datasets', train=False, transform=to_bernoulli, download=True),
    batch_size=10000, shuffle=True)

for (data, label) in test_loader:
    test_tensor_list = (data.cuda(async=False).view(-1, 784), label.cuda(async=False))
class myiterator:
    def __iter__(self):
        return iter([test_tensor_list])
#new_test_loader = myiterator()

#hps = {'zdim:', zdim}
#hps = {'x_dim':784,'z_dim':args.zdim,'hidden_dim':400}
hps = {'z_dim':args.zdim, 'noise_dim':args.zdim, 'dim':784}

def main(f=args.file):
    #hps = get_default_hparams()
    #model = VAE(hps)
    if args.model == 'VAE':
        model = VAE(**hps)
    if args.model == 'BAE':
        model = BAE(**hps)

    model.cuda()
    try:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage)['state_dict'])
    except:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))
    model.eval()

    #loader = simulate_data(model, batch_size=100, n_batch=10)
    loader = myiterator()
    forward_ais(model, loader, forward_schedule=np.linspace(0., 1., args.num_steps), n_sample=args.num_samples)


if __name__ == '__main__':
    main()