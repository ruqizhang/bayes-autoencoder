import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
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

    def __init__(self, x_dim,z_dim,hidden_dim,vocab_size,dropout,bsz,device_id):
        super(Decode, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.fc5 = nn.Linear(z_dim+2,hidden_dim)
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

    def forward(self,x_emb, z,y):
        c0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        z_y = torch.cat((z,y),dim = 1)
        h0 = self.fc5(z_y)
        h0 = h0.unsqueeze(0)
        # h0 = Variable(torch.zeros((1,self.bsz,self.hidden_dim)).cuda(self.device_id))
        s0 = (h0,c0)
        ht,st = self.lstm(x_emb,s0)
        ht = self.drop(ht)
        recon_batch = self.fc4(ht)
        return recon_batch

class Label(nn.Module):

    def __init__(self,x_dim,z_dim,hidden_dim,vocab_size,dropout):
        super(Label, self).__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        #self.hw1 = Highway(x_dim, 4, F.relu)
        self.lstm = nn.LSTM(x_dim, hidden_dim,dropout=dropout)
        # self.hw1 = Highway(hidden_dim, 1, F.relu)
        self.fc1 = nn.Linear(hidden_dim, 2)
        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        # print(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[-1,:,:]
        lstm_out = self.drop(lstm_out)
        # lstm_out = lstm_out[:, :self.hidden_dim] + lstm_out[: ,self.hidden_dim:]
        # print(lstm_out)
        # lstm_out = lstm_out.contiguous().view(-1,seq_len*hidden_dim)
        recon_label = self.fc1(lstm_out)
        probs = F.softmax(recon_label,dim = 1)
        # print(recon_label)

        return probs

class VAE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, z_dim,nlayers, device_id, bsz,dropout=0.5, tie_weights=False):
        super(VAE, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(ntoken, ninp)
        self.encoder = Encode(ninp,z_dim,nhid,ntoken,dropout)
        self.decoder = Decode(ninp,z_dim,nhid,ntoken,dropout,bsz,device_id)
        self.label = Label(ninp,z_dim,nhid,ntoken,dropout)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.device_id = device_id
        self.bsz = bsz
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)


    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          eps.cuda(self.device_id)
          return eps.mul(std).add_(mu)
        else:
          return mu
    def label_recon(self,x):
        fake_label = self.label(x)

        return fake_label
    def forward(self, input,label_list = None):
        emb = self.drop(self.word_embeddings(input))
        mu,logvar = self.encoder(emb)
        fake_label = self.label_recon(emb)
        z = self.reparameterize(mu, logvar)
        if label_list is not None:
            recon_batch = self.decoder(emb,z,label_list)
        else:
            recon_batch = self.decoder(emb,z,fake_label)
        return recon_batch, mu,logvar,fake_label
