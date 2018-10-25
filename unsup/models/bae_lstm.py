import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

__all__ = ['BAE_LSTM']

#TODO: update encoder + decoder to actually use noise dim

class Encode(nn.Module):

    def __init__(self,x_dim,z_dim,noise_dim,hidden_dim,vocab_size,dropout,bsz,device_id=None):
        super(Encode, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.noise_dim = noise_dim

        self.hidden_dim = hidden_dim
        self.bsz = bsz
        #self.device_id = device_id
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        self.fc21 = nn.Linear(hidden_dim, z_dim) #mean
        # self.fc22 = nn.Linear(hidden_dim, z_dim) #logvar
        self.drop = nn.Dropout(dropout)
        self.fc5 = nn.Linear(z_dim,hidden_dim)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.fc21.bias.data.fill_(0)
        self.fc21.weight.data.uniform_(-initrange, initrange)
        self.fc5.bias.data.fill_(0)
        self.fc5.weight.data.uniform_(-initrange, initrange)
    """def noise(self):
        #xi = Variable(torch.randn(self.bsz,self.z_dim).cuda(self.device_id),requires_grad=True)
        xi = torch.normal(self.bsz, self.z_dim )
        return xi"""

    def forward(self, x):
        #c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        c0 = torch.zeros((1, self.bsz, self.hidden_dim), device = x.device, dtype = x.dtype)
        #xi = self.noise()
        xi = torch.randn(self.bsz, self.z_dim, device = x.device, dtype = x.dtype, requires_grad = True)
        h0 = self.fc5(xi)
        h0 = h0.unsqueeze(0)
        # h0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        s0 = (h0,c0)
        lstm_out, _ = self.lstm(x,s0)
        lstm_out = lstm_out[-1,:,:]
        lstm_out = self.drop(lstm_out)
        z = self.fc21(lstm_out)
        return z,xi

class Decode(nn.Module):

    def __init__(self, x_dim,z_dim,noise_dim,hidden_dim,vocab_size,dropout,bsz,device_id=None):
        super(Decode, self).__init__()
        self.z_dim = z_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.fc5 = nn.Linear(z_dim,hidden_dim)
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        self.fc4 = nn.Linear(hidden_dim,vocab_size)
        self.drop = nn.Dropout(dropout)
        self.bsz = bsz
        self.device_id = device_id
        self.vocab_size = vocab_size
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.fc5.bias.data.fill_(0)
        self.fc5.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.fill_(0)
        self.fc4.weight.data.uniform_(-initrange, initrange)

    def forward(self,x_emb, z):
        #c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        c0 = torch.zeros((1, self.bsz, self.hidden_dim), device = x_emb.device, dtype = x_emb.dtype)
        h0 = self.fc5(z)
        h0 = h0.unsqueeze(0)
        # h0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        s0 = (h0,c0)
        ht,st = self.lstm(x_emb,s0)
        ht = self.drop(ht)
        recon_batch = self.fc4(ht)
        return recon_batch.view(-1,self.vocab_size)

class baeLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhidden, zdim, noise_dim, nlayers, bsz=64,dropout=0.5, tie_weights=False):
        super(baeLSTM, self).__init__()
        self.ntoken = ntoken
        self.zdim = zdim

        self.drop = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(ntoken, ninp)

        self.encoder = Encode(ninp,zdim,noise_dim,nhidden,ntoken,dropout,bsz)
        self.decoder = Decode(ninp,zdim,noise_dim,nhidden,ntoken,dropout,bsz)
        
        self.nhid = nhidden
        self.nlayers = nlayers
        self.device_id = None
        self.bsz = bsz
        
        self.embed = nn.Sequential(
            self.word_embeddings,
            self.drop)
        
        self.init_weights()
        
    
    def init_weights(self):
        initrange = 0.1
        #self.zdim = zdim
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, z = None):
        emb = self.embed(input)

        if z is None:
            z, noise = self.encoder(emb)
        else:
            noise = None
        recon_batch = self.decoder(emb,z)

        return recon_batch,z, noise

    def prior_loss(self,prior_std):
        prior_loss = 0.0
        for var in self.parameters():
            prior_dist = torch.distributions.Normal(torch.zeros_like(var), prior_std * torch.ones_like(var))
            prior_loss += -prior_dist.log_prob(var).sum()
            #nn = torch.div(var, prior_std)
            #prior_loss += torch.sum(nn*nn)
        #print('prior_loss',prior_loss)#1e-3
        return 0.5*prior_loss

    def noise_loss(self,lr,alpha):
        noise_loss = 0.0
        noise_std = (2.0 * lr * alpha)**0.5
        for var in self.parameters():
            rand_like_var = torch.zeros_like(var).normal_() * noise_std
            noise_loss += torch.sum(var * rand_like_var)
        #print('noise_loss',noise_loss)#1e-8
        return noise_loss

    def generate(self, decoder_input, decoder_hidden):
        emb = self.word_embeddings(decoder_input)
        decoder_output, decoder_hidden = self.decoder.lstm(emb.unsqueeze(0), decoder_hidden)
        decoder_output = self.decoder.fc4(decoder_output[-1])

        return decoder_output, decoder_hidden

    def criterion(self, recon, data, target, reduction='elementwise_mean'):
        recon = recon.view(-1, self.ntoken)

        return torch.nn.functional.cross_entropy(recon, target, reduction=reduction)

class BAE_LSTM:
    args = list()
    kwargs = {}
    transform_train, transform_test = None, None
    base = baeLSTM
