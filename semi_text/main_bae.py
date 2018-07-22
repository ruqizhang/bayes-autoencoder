# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import model_bae
import torch.optim as optim
import numpy as np
import classification_datasets
from torchtext import data
import itertools
parser = argparse.ArgumentParser(description='PyTorch Semisup MR')
parser.add_argument('--numlabel',type = int,default=2000, help = 'number of labeled data')
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

model = model_bae.VAE(args.model, ntokens, args.emsize, args.nhid, args.zdim,args.nlayers,device_id,args.batch_size, args.dropout, args.tied)
if args.cuda:
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

def loss_function(recon_batch, x):
    # print(recon_batch,x)
    recon_batch = recon_batch.view(-1,ntokens)
    BCE = F.cross_entropy(recon_batch, x)
    return BCE

def en_loss(z_recon,z):
    z = Variable(z.data,requires_grad = False)
    loss = F.mse_loss(z_recon,z)
    return loss

def z_prior_loss(z):
    prior_loss = 0.5*torch.sum(z*z)
    return prior_loss

def z_noise_loss(z):

    learning_rate = lr

    noise_std = np.sqrt(2*learning_rate*alpha)
    noise_std = torch.from_numpy(np.array([noise_std])).float().cuda(device_id)
    noise_std = noise_std[0]
    means = torch.zeros(z.size()).cuda(device_id)
    noise_loss = torch.sum(z * Variable(torch.normal(means, std = noise_std).cuda(device_id),
                           requires_grad = False))
    #print('noise_loss',noise_loss)#1e-8
    return noise_loss
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
        model.encoder.bsz = data.size(1)
        model.label.bsz = data.size(1)
        out_ix = data[1:,:].contiguous().view(-1)
        row = range(args.batch_size)
        label_2 = Variable(torch.zeros(args.batch_size,2).cuda(device_id),requires_grad = False)
        label_2[row,label] = 1
        recon_batch, z,fake_label = model(data[:-1,:])
        _,pred_label = torch.max(fake_label,1)
        pred_res += list(pred_label.data)
        count+=1
    acc = get_accuracy(truth_res,pred_res)
    print(' acc :%g ' % (acc ))
    return acc

def z_opt(z_sample):

    opt = optim.SGD([z_sample], lr=lr, momentum = 1-alpha)

    return opt
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
        args.bptt = (data.size(0)+undata.size(0))/2
        out_ix = data[1:,:].contiguous().view(-1)
        unout_ix = undata[1:,:].contiguous().view(-1)
        row = range(data.size(1))
        label_2 = Variable(torch.zeros(data.size(1),2).cuda(device_id),requires_grad = False)
        label_2[row,label] = 1
        model.zero_grad()
        for j in range(J):
            if j == 0:
                model.zero_grad()
                model.decoder.bsz = data.size(1)
                model.encoder.bsz = data.size(1)
                model.label.bsz = data.size(1)
                recon_batch,z,fake_label = model(data[:-1,:])
                model.decoder.bsz = undata.size(1)
                model.encoder.bsz = undata.size(1)
                model.label.bsz = undata.size(1)
                unrecon_batch,unz,_ = model(undata[:-1,:])
                z_sample = Variable(z.data,requires_grad = True)
                z_optimizer = z_opt(z_sample)
                z_optimizer.zero_grad()
                unz_sample = Variable(unz.data,requires_grad = True)
                unz_optimizer = z_opt(unz_sample)
                unz_optimizer.zero_grad()
            else:
                model.zero_grad()
                emb = model.embed(data[:-1,:])
                model.decoder.bsz = data.size(1)
                model.label.bsz = data.size(1)
                fake_label = model.label(emb,z_sample)
                recon_batch = model.decoder(emb,z_sample)
                model.decoder.bsz = undata.size(1)
                model.label.bsz = undata.size(1)
                unemb = model.embed(undata[:-1,:])
                # unfake_label = model.label(unemb,unz_sample)
                unrecon_batch = model.decoder(unemb,unz_sample)

            BCE = loss_function(recon_batch, out_ix)
            unBCE = loss_function(unrecon_batch, unout_ix)
            label_loss = loss_label(fake_label,label_2)
            prior_loss = model.prior_loss(prior_std)
            noise_loss = model.noise_loss(lr,alpha)
            prior_loss /=args.bptt*len(train_data)
            noise_loss /=args.bptt*len(train_data)
            prior_loss_z = z_prior_loss(z_sample)
            noise_loss_z = z_noise_loss(z_sample)
            prior_loss_z /=args.bptt*len(train_data)
            noise_loss_z /=args.bptt*len(train_data)
            unprior_loss_z = z_prior_loss(unz_sample)
            unnoise_loss_z = z_noise_loss(unz_sample)
            unprior_loss_z /=args.bptt*len(train_data)
            unnoise_loss_z /=args.bptt*len(train_data)
            loss = BCE+unBCE+label_loss+ prior_loss+noise_loss+prior_loss_z+noise_loss_z+unprior_loss_z+unnoise_loss_z

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            if j>burnin:
                z_optimizer.step()
                unz_optimizer.step()
                model.zero_grad()
                loss_en = en_loss(z_sample,z)
                unloss_en = en_loss(unz_sample,unz)
                loss = loss_en+unloss_en
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                optimizer.step()
        count += 1

        total_loss += label_loss.data+BCE.data+unBCE.data
        _,pred_label = torch.max(torch.exp(fake_label),1)
        pred_res += list(pred_label.data)
        if count % args.log_interval == 0 and count > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | lr {:5.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}  '.format(
                epoch,   lr,
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
    print('epoch: %d done!\n acc:%g'%(epoch, get_accuracy(truth_res,pred_res)))

# Loop over epochs.
lr = args.lr
alpha = 0.7
prior_std = 1
burnin = 0
J = 2
optimizer = optim.SGD(model.parameters(), lr=lr,momentum = 1-alpha)
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    # val_loss = evaluate(val_data)
    test_loss = evaluate(test_data)
    if epoch > 80:
        print('save!')
        torch.save(model.state_dict(),'./checkpoints/model_un1000_a0.7_2%i%i.pt'%(mt,epoch))
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test acc{:5.5f} | '
            .format(epoch, (time.time() - epoch_start_time),
                                       test_loss))
