import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
class Encode(nn.Module):

    def __init__(self,x_dim,z_dim,hidden_dim,vocab_size,dropout,bsz,device_id):
        super(Encode, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.bsz = bsz
        self.device_id = device_id
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        self.fc21 = nn.Linear(hidden_dim, z_dim) #mean
        self.drop = nn.Dropout(dropout)
        self.fc5 = nn.Linear(z_dim,hidden_dim)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.fc21.bias.data.fill_(0)
        self.fc21.weight.data.uniform_(-initrange, initrange)
        self.fc5.bias.data.fill_(0)
        self.fc5.weight.data.uniform_(-initrange, initrange)
    def noise(self):
        xi = Variable(torch.randn(self.bsz,self.z_dim).cuda(self.device_id))
        return xi

    def forward(self, x):
        c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        xi = self.noise()
        h0 = self.fc5(xi)
        h0 = h0.unsqueeze(0)
        s0 = (h0,c0)
        lstm_out, _ = self.lstm(x,s0)
        lstm_out = lstm_out[-1,:,:]
        lstm_out = self.drop(lstm_out)
        z = self.fc21(lstm_out)
        return z

class Decode(nn.Module):

    def __init__(self, x_dim,z_dim,hidden_dim,vocab_size,dropout,bsz,device_id):
        super(Decode, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc5 = nn.Linear(z_dim,hidden_dim)
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        self.fc4 = nn.Linear(hidden_dim,vocab_size)
        self.drop = nn.Dropout(dropout)
        self.bsz = bsz
        self.device_id = device_id
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.fc5.bias.data.fill_(0)
        self.fc5.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.fill_(0)
        self.fc4.weight.data.uniform_(-initrange, initrange)

    def forward(self,x_emb, z):
        c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        h0 = self.fc5(z)
        h0 = h0.unsqueeze(0)
        s0 = (h0,c0)
        ht,st = self.lstm(x_emb,s0)
        ht = self.drop(ht)
        recon_batch = self.fc4(ht)
        return recon_batch
class Label(nn.Module):

    def __init__(self,x_dim,z_dim,hidden_dim,vocab_size,dropout,bsz,device_id):
        super(Label, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.fc5 = nn.Linear(z_dim,hidden_dim)
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 2)
        self.drop = nn.Dropout(dropout)
        self.device_id = device_id
        self.bsz = bsz
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.fc5.bias.data.fill_(0)
        self.fc5.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-initrange, initrange)

    def forward(self, x,z):
        c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        h0 = self.fc5(z)
        h0 = h0.unsqueeze(0)
        s0 = (h0,c0)
        lstm_out, _ = self.lstm(x,s0)
        lstm_out = lstm_out[-1,:,:]
        lstm_out = self.drop(lstm_out)
        recon_label = self.fc1(lstm_out)
        probs = F.softmax(recon_label,dim = 1)

        return probs
class VAE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, z_dim,nlayers, device_id, bsz,dropout=0.5, tie_weights=False):
        super(VAE, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(ntoken, ninp)
        self.encoder = Encode(ninp,z_dim,nhid,ntoken,dropout,bsz,device_id)
        self.decoder = Decode(ninp,z_dim,nhid,ntoken,dropout,bsz,device_id)
        self.label = Label(ninp,z_dim,nhid,ntoken,dropout,bsz,device_id)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.device_id = device_id
        self.bsz = bsz
        self.embed = nn.Sequential(
            self.word_embeddings,
            self.drop)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input):
        emb = self.embed(input)
        z = self.encoder(emb)
        fake_label = self.label(emb,z)
        recon_batch = self.decoder(emb,z)
        return recon_batch,z,fake_label

    def noise_loss(self,lr,alpha):
        noise_loss = 0.0
        noise_std = np.sqrt(2/lr*alpha)
        for var in self.parameters():
            means = torch.zeros(var.size()).cuda(self.device_id)
            noise = Variable(torch.normal(means, std = noise_std).cuda(self.device_id),requires_grad = False)
            noise_loss += torch.sum(var * noise)
        return noise_loss
  