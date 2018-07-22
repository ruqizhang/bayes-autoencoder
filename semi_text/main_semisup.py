# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import models
import torch.optim as optim
import classification_datasets
from torchtext import data
import itertools
parser = argparse.ArgumentParser(description='PyTorch LSTM MR Semisup')
parser.add_argument('--numlabel',type = int,default=2000, help = 'number of labeled data')
parser.add_argument('--model', type=str, default='bvae',
                    help='type of recurrent net (vae, bvae)')
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
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--device_id',type = int, help = 'device id to use')
args = parser.parse_args()
device_id = args.device_id
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
args.cuda = True
###############################################################################
# Load data
###############################################################################

text_field = data.Field(lower=True,init_token = '<SOS>',eos_token='<EOS>')
label_field = data.Field(sequential=False)

unsup_data, train_data,  test_data = classification_datasets.load_mr_semi(text_field, label_field, numlabel = args.numlabel,batch_size=args.batch_size)
ntokens=len(text_field.vocab)
iterator = zip(unsup_data, itertools.cycle(train_data))

###############################################################################
# Build the model
###############################################################################
print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)
model = model_cfg.VAE("LSTM", ntokens, args.emsize, args.nhid, args.zdim,args.nlayers,device_id,args.batch_size, args.dropout, args.tied)
model.cuda(device_id)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def loss_function(recon_batch, x, mu, logvar):
    recon_batch = recon_batch.view(-1,ntokens)
    BCE = F.cross_entropy(recon_batch, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= args.batch_size*args.bptt
    return BCE, KLD
def loss_label(fake_label,label_2):

    loss = F.binary_cross_entropy(fake_label,label_2)

    return loss
def get_accuracy(truth, pred):
    assert len(truth)==len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i]==pred[i]:
             right += 1.0
    return right/len(truth)
def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_kld = 0
    count=0
    truth_res = []
    pred_res = []
    # hidden = model.init_hidden(eval_batch_size)
    for batch in data_source:
        data, label = batch.text, batch.label
        data,label = data.cuda(device_id), label.cuda(device_id)
        label.data.sub_(2)

        truth_res += list(label.data)
        args.batch_size = data.size(1)
        model.decoder.bsz = args.batch_size
        seq_len = data.size(0)-1
        out_ix = data[1:,:].contiguous().view(-1)
        row = range(args.batch_size)
        label_2 = Variable(torch.zeros(args.batch_size,2).cuda(device_id),requires_grad = False)
        label_2[row,label] = 1
        recon_batch, mu,logvar,fake_label = model(data[:-1,:],label_2)
        BCE,KLD = loss_function(recon_batch, out_ix,mu,logvar)
        loss = BCE+KLD
        _,pred_label = torch.max(fake_label,1)
        pred_res += list(pred_label.data)
        total_loss += loss.data[0]
        total_kld += KLD.data[0]
        count+=1
    avg = total_loss / count
    avg_kld = total_kld/count
    acc = get_accuracy(truth_res,pred_res)
    print(' acc :%g avg_loss:%g kld:%g' % (acc,avg, avg_kld ))
    return acc


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    model.decoder.bsz = args.batch_size
    truth_res = []
    pred_res = []
    count = 0.0
    # hidden = model.init_hidden(args.batch_size)
    for (unbatch, lbatch) in iterator:
        data, label = lbatch.text, lbatch.label
        undata = unbatch.text
        undata = undata.cuda(device_id)
        data,label = data.cuda(device_id), label.cuda(device_id)
        data.volatile = False
        label.volatile = False
        label.data.sub_(2)
        truth_res += list(label.data)
        args.batch_size = data.size(1)
        model.decoder.bsz = args.batch_size
        seq_len = data.size(0)-1
        out_ix = data[1:,:].contiguous().view(-1)
        unout_ix = undata[1:,:].contiguous().view(-1)
        row = range(args.batch_size)
        label_2 = Variable(torch.zeros(args.batch_size,2).cuda(device_id),requires_grad = False)
        label_2[row,label] = 1
        model.zero_grad()
        recon_batch, mu,logvar,fake_label = model(data[:-1,:],label_2)
        BCE,KLD = loss_function(recon_batch, out_ix,mu,logvar)
        label_loss = loss_label(fake_label,label_2)
        loss = label_loss + BCE + KLD

        model.decoder.bsz = undata.size(1)
        recon_batch, mu,logvar,_ = model(undata[:-1,:])
        unBCE,unKLD = loss_function(recon_batch, unout_ix,mu,logvar)
        loss += unBCE + unKLD
        if args.model == "bvae":
            prior_loss = model.prior_loss(prior_std)
            noise_loss = model.noise_loss(lr,alpha)
            prior_loss /=args.bptt*len(train_data)
            noise_loss /=args.bptt*len(train_data)
            loss += prior_loss+noise_loss
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        count +=1
        total_loss += loss.data
        _,pred_label = torch.max(torch.exp(fake_label),1)
        pred_res += list(pred_label.data)
        if count % args.log_interval == 0 and count > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | lr {:5.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}  | kld {:5.9f}'.format(
                epoch,   lr,
                elapsed * 1000 / args.log_interval, cur_loss, KLD.data[0]))
            total_loss = 0
            start_time = time.time()
    print('epoch: %d done!\n acc:%g'%(epoch, get_accuracy(truth_res,pred_res)))
# Loop over epochs.
lr = args.lr
alpha = 0.7
prior_std = 1
optimizer = optim.SGD(model.parameters(), lr=lr,momentum = 1-alpha)
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    test_loss = evaluate(test_data)
    if epoch > 80:
        print('save!')
        torch.save(model.state_dict(),'./checkpoints/model_un1500_a0.7_3_seed1%i.pt'%(epoch))
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test acc {:5.5f} | '
            .format(epoch, (time.time() - epoch_start_time),
                                       test_loss))