# coding: utf-8
import argparse
import time
import math,os,sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import data
import models
#from models.bae import BAE
import torch.optim as optim
import numpy as np
from itertools import starmap, cycle

import utils

from torch.autograd.gradcheck import zero_gradients
parser = argparse.ArgumentParser(description='PyTorch PTB RNN/LSTM Language Model')
parser.add_argument('--dataset', type=str, default='ptb', 
                    help = 'dataset to use (only ptb is implemented currently)')
parser.add_argument('--data_path', type=str, default='./datasets/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--zdim', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--alpha', type=float, default=0.1, help='1-momentum (default: 0.1)')
parser.add_argument('--burnin', type=int, default=0,
                    help='number of burnin steps (default 0)')
parser.add_argument('--J', type=int, default=2, 
                    help='number of updates per mini-batch (default for sim. 2, use 4 for gibbs)')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--gibbs', action='store_true',
                    help='gibbs updating or simulatneous updating')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--save_epochs', type = int, default = 10, help = 'how often to save the model')
parser.add_argument('--dir', type = str, default = 'directory to save model to')
parser.add_argument('--use_test', action='store_true', 
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
args.cuda = True

if not args.gibbs:
    print('Using simulataneous updating')
    args.gibbs = False
else:
    print('Using Gibbs updating')

#make directory
print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir+'/results/', exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write('python ' + ' '.join(sys.argv))
    f.write('\n')


###############################################################################
#define the model
###############################################################################
print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

###############################################################################
# Load data
###############################################################################
loaders, ntokens = data.loaders(args.dataset, args.data_path, args.batch_size, args.bptt, 
                        model_cfg.transform_train, model_cfg.transform_test,
                        use_validation=not args.use_test, use_cuda=args.cuda)

###############################################################################
# Build the model
###############################################################################
print('Preparing model')
print(*model_cfg.args)
print('using ', args.zdim, ' latent space')
model = model_cfg.base(*model_cfg.args, 
                    noise_dim=args.zdim, zdim=args.zdim, 
                    ntoken=ntokens, ninp=args.emsize, nhidden=args.nhid, 
                    nlayers=args.nlayers, bsz=args.batch_size, 
                    dropout=args.dropout, tie_weights=args.tied,
                    **model_cfg.kwargs)
model.cuda()

# Loop over epochs.
#lr = args.lr
#alpha = 0.1
bit = 0
prior_std = 1
epoch_cut = 0
#burnin = 0
#J = 2
best_val_loss = 1e10
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum = 1.-args.alpha)

start_epoch = 1
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(start_epoch, args.epochs+1):
        epoch_start_time = time.time()

        #train
        utils.train(epoch, loaders['train'], model, optimizer, ntokens, \
                    args.lr, args.alpha, args.J, args.burnin, prior_std, args.clip, \
                    args.log_interval, args.bptt, \
                    gibbs = args.gibbs)

        #validate
        with torch.no_grad():
            if 'valid' in loaders:
                val_loss = utils.evaluate(loaders['valid'], model, ntokens, epoch, args.dir)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                val_loss, math.exp(val_loss)))
                print('-' * 89)

                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    print('New Best!')
                    best_val_loss = val_loss

            if 'test' in loaders:
                #test
                test_loss = utils.evaluate(loaders['test'], model, ntokens, epoch, args.dir)

                print('-' * 89)
                try:
                    test_ppl = math.exp(test_loss)
                except:
                    test_ppl = 1e10
                print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                        'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                test_loss, test_ppl))
                print('-' * 89)


            if epoch % args.save_epochs is 0:
                print('saving model')
                utils.save_model(epoch, model, optimizer, args.dir)
            
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')