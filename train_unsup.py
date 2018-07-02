from __future__ import print_function
import argparse
import torch
import torch.utils.data
import os, sys
import data, models, utils

parser = argparse.ArgumentParser(description='VAE MNIST')
parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name (default: MNIST)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs to train (default: 3000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--K', type=int, default=1, metavar='N',
                    help='how many samples to draw in the IWAE loss')
parser.add_argument('--alpha', type=float, default=1, metavar='N',
                    help='what value of alpha to draw in the variational Renyi loss (default: 1)')
parser.add_argument('--save-epochs', type=int, default=25, metavar='N',
                    help='how often to save the model')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--zdim', type=int, default = 50, metavar = 'S',
                    help='latent + noise dimension to use in model')
parser.add_argument('--optimizer', type=str, default='Adam',
                    help='which optimizer to use')
parser.add_argument('--optimizer_options', nargs='*',
                    help = 'additional options for optimizer')
parser.add_argument('--dir', type=str, default='/scratch/wm326/vae', help='output directory to store')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--bvae', action='store_true', help='whether to use the bvae instead of vae')
args = parser.parse_args()

#set cuda if available, while also setting seed
torch.manual_seed(args.seed)
args.device = None
if torch.cuda.is_available() and not args.no_cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
    
#define the model
print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Preparing model')
print(*model_cfg.args)
print('using ', args.zdim, ' latent space')
model = model_cfg.base(*model_cfg.args, zdim=args.zdim, **model_cfg.kwargs)
model.to(args.device)
model.device = args.device

#prepare the dataset
print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    unsup=True
)
#prepare the optimizer
optimizer = utils.construct_optimizer(model.parameters(), args.optimizer, args.optimizer_options)

#make directory
print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir+'/results/', exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write('python '.join(sys.argv))
    f.write('\n')

#create loss function - note K determines importance weighting, alpha renyi scaling/pvi scaling
#default is to use the variational renyi bound with alpha = 1.0 and K = 1
def criterion(data, K = args.K, alpha=args.alpha):
    std_loss = utils.calculate_loss(model, data, K=K, alpha=alpha)
    if args.bvae:
        prior_loss = 0.0
        for param in model.parameters():
            prior_dist = torch.distributions.Normal(torch.zeros_like(param), torch.ones_like(param))
            prior_loss -= prior_dist.log_prob(param).sum()

        return std_loss + prior_loss/data.size(0)
    else:
        return std_loss

training_loss = [None]*(args.epochs + 1)
testing_loss = [None]*(args.epochs + 1)

for epoch in range(1, args.epochs + 1):
    training_loss[epoch] = utils.train(model, optimizer, loaders['train'], criterion, args.device, epoch=epoch, log_interval=args.log_interval)
    
    with torch.no_grad():
        save_recon = False
        kwargs = {}
        if epoch%args.save_epochs is 0:
            print('saving models and images at ', epoch)
            utils.save_model(epoch, model, optimizer, args.dir)
            save_recon = True

            kwargs = {'dir': args.dir +'/results', 'epoch':epoch}

        K = 10
        print('Using ', K, 'samples for testing LL')
        testing_loss[epoch] = utils.test(model, loaders['test'], criterion, K, args.device, save_recon, **kwargs)




