import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

__all__ = ['BAE']

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
    def noise(self):
        xi = Variable(torch.randn(self.bsz,self.z_dim).cuda(self.device_id),requires_grad=True)
        return xi

    def forward(self, x):
        c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        xi = self.noise()
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
        # h0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        s0 = (h0,c0)
        ht,st = self.lstm(x_emb,s0)
        ht = self.drop(ht)
        recon_batch = self.fc4(ht)
        return recon_batch

class BAE_LSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, zdim, nlayers, device_id, bsz,dropout=0.5, tie_weights=False):
        super(BAE_LSTM, self).__init__()
        self.zdim = zdim

        self.drop = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(ntoken, ninp)

        self.encoder = Encode(ninp,zdim,nhid,ntoken,dropout,bsz,device_id)
        self.decoder = Decode(ninp,zdim,nhid,ntoken,dropout,bsz,device_id)
        #self.decode = self.decode

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
        #self.zdim = zdim
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)

    def decode(self, z, emb):
        return self.decoder(emb, z)
    def forward(self, input):
        emb = self.embed(input)
        z,xi = self.encoder(emb)
        recon_batch = self.decoder(emb,z)
        return recon_batch,z,xi

    def prior_loss(self,prior_std):
        prior_loss = 0.0
        for var in self.parameters():
            nn = torch.div(var, prior_std)
            prior_loss += torch.sum(nn*nn)
        #print('prior_loss',prior_loss)#1e-3
        return 0.5*prior_loss

    def noise_loss(self,lr,alpha):
        noise_loss = 0.0
        # learning_rate = base_lr * np.exp(-lr_decay *min(1.0, (train_iter*args.batch_size)/float(datasize)))
        learning_rate = lr
        noise_std = np.sqrt(2*learning_rate*alpha)
        noise_std = torch.from_numpy(np.array([noise_std])).float().cuda(self.device_id)
        noise_std = noise_std[0]
        for var in self.parameters():
            means = torch.zeros(var.size()).cuda(self.device_id)
            noise_loss += torch.sum(var * Variable(torch.normal(means, std = noise_std).cuda(self.device_id),
                               requires_grad = False))
        #print('noise_loss',noise_loss)#1e-8
        return noise_loss

    def generate(self, decoder_input, decoder_hidden):
        emb = self.word_embeddings(decoder_input)
        decoder_output, decoder_hidden = self.decoder.lstm(emb.unsqueeze(0), decoder_hidden)
        decoder_output = self.decoder.fc4(decoder_output[-1])

        return decoder_output, decoder_hidden

class BAE:
    args = list()
    kwargs = {'dropout':0.5, 'tie_weights':False}
    base = BAE_LSTM