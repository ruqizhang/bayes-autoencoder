import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['VAE_LSTM']

class Encode(nn.Module):

    def __init__(self,x_dim,z_dim,hidden_dim,vocab_size,dropout):
        super(Encode, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        self.fc21 = nn.Linear(hidden_dim, z_dim) #mean
        self.fc22 = nn.Linear(hidden_dim, z_dim) #logvar
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.fc21.bias.data.fill_(0)
        self.fc21.weight.data.uniform_(-initrange, initrange)
        self.fc22.bias.data.fill_(0)
        self.fc22.weight.data.uniform_(-initrange, initrange)
    # def init_hidden(self):
    #     # the first is the hidden h
    #     # the second is the cell  c
    #     return (Variable(torch.zeros(1, args.batch_size, self.hidden_dim).cuda(device_id)),
    #             Variable(torch.zeros(1, args.batch_size, self.hidden_dim).cuda(device_id)))

    def forward(self, x):
        # h = self.init_hidden()
        # x = x.view(args.batch_size , x.size(1), self.x_dim)

        #print(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[-1,:,:]
        lstm_out = self.drop(lstm_out)
        mu = self.fc21(lstm_out)
        logvar = self.fc22(lstm_out)
        return mu, logvar

class Decode(nn.Module):

    def __init__(self, x_dim,z_dim,hidden_dim,vocab_size,dropout,bsz):
        super(Decode, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc5 = nn.Linear(z_dim,hidden_dim)
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        self.fc4 = nn.Linear(hidden_dim,vocab_size)
        self.drop = nn.Dropout(dropout)
        self.bsz = bsz
        #self.device_id = device_id
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.fc5.bias.data.fill_(0)
        self.fc5.weight.data.uniform_(-initrange, initrange)
        self.fc4.bias.data.fill_(0)
        self.fc4.weight.data.uniform_(-initrange, initrange)

    def forward(self,x_emb, z):
        #c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        c0 = torch.zeros((1, self.bsz, self.hidden_dim), device = z.device, dtype = z.dtype)
        h0 = self.fc5(z)
        h0 = h0.unsqueeze(0)
        # h0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        s0 = (h0,c0)
        ht,st = self.lstm(x_emb,s0)
        ht = self.drop(ht)
        recon_batch = self.fc4(ht)
        return recon_batch

class vaeLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhidden, zdim, noise_dim, nlayers, bsz,dropout=0.5, tie_weights=False):
        super(vaeLSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(ntoken, ninp)
        self.encoder = Encode(ninp,zdim,nhidden,ntoken,dropout)
        self.decoder = Decode(ninp,zdim,nhidden,ntoken,dropout,bsz)
        self.z_dim = zdim
        self.nhid = nhidden
        self.nlayers = nlayers
        #self.device_id = device_id
        self.bsz = bsz
        self.ntoken = ntoken
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)


    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          #eps = Variable(std.data.new(std.size()).normal_())
          eps = torch.zeros_like(std).normal_()
          #eps.cuda(self.device_id)
          return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, input, z = None):
        emb = self.drop(self.word_embeddings(input))
        mu,logvar = self.encoder(emb)

        if z is None:
            z = self.reparameterize(mu, logvar)
            recon_batch = self.decoder(emb,z)
            return recon_batch, mu,logvar
        else:
            recon_batch = self.decoder(emb, z)
            return recon_batch, z, None

    def criterion(self, recon, data, target, reduction='elementwise_mean'):
        recon = recon.view(-1, self.ntoken)

        return torch.nn.functional.cross_entropy(recon, target,reduction=reduction)

class VAE_LSTM:
    args = list()
    kwargs = {}
    transform_train, transform_test = None, None
    base = vaeLSTM
