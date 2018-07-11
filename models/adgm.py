import torch
from torch import nn
from torchvision.utils import save_image
from torchvision import transforms
import torch.distributions

from .mlp import MLP, Linear2
import inferences
from LogSumExp import LogSumExp

__all__=['ADGM', 'BADGM']

def compute_prior_loss(model, scale = 1.0):
    loss = 0.0
    for param in model.parameters():
        param_dist = torch.distributions.Normal(torch.zeros_like(param), scale * torch.ones_like(param))
        loss -= param_dist.log_prob(param).sum()

    return loss

class baseADGM(nn.Module):
    #heavily based on https://github.com/wohlert/semi-supervised-pytorch
    def __init__(self, dim = 784, nclasses =10, zdim = 50, adim = 50, hidden = 500, activation = nn.ReLU):
        super(baseADGM, self).__init__()
        self.xdim, self.nclasses, self.zdim, self.adim, self.hidden = dim, nclasses, zdim, adim, hidden
        
        self.aux_encoder = MLP(self.xdim, hidden, adim, out_layer = Linear2, activation = activation)
        self.aux_decoder = MLP(self.xdim + zdim + nclasses, hidden, adim, out_layer = Linear2, activation = activation)
        
        self.encoder_y_real = MLP(self.xdim + adim, hidden, nclasses, activation = activation)
        self.encoder = MLP(adim + nclasses + self.xdim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(nclasses + zdim, hidden, self.xdim)
        
        self.norm_classifier_outputs = nn.LogSoftmax(dim=1)
        
        
    def classify(self, x, a):            
        #classifier q(y|a,x)
        logits_real = self.encoder_y_real(torch.cat([x, a],dim=1))
        logits = self.norm_classifier_outputs(logits_real)
        return logits
    
    def forward(self, x, y):
        x, y = x.view(-1, self.xdim), y.view(-1, self.nclasses)
        
        #auxiliary inference q(a|x)
        amu, alogvar = self.aux_encoder(x)
        
        qa_dist = torch.distributions.Normal(amu, torch.exp(alogvar.mul(0.5)))
        if self.train:
            qa = qa_dist.rsample()
        else:
            qa = amu
        
        #classifier q(y|a,x)
        logits = self.classify(x, qa)
        
        # Latent inference q(z|a,y,x)
        zmu, zlogvar = self.encoder(torch.cat([x, y, qa],dim=1))
        qz_dist = torch.distributions.Normal(zmu, torch.exp(zlogvar.mul(0.5)))
        
        if self.train:
            qz = qz_dist.rsample()
        else:
            qz = zmu
            
        # Generative p(x|z,y)
        x_real = self.decoder(torch.cat([qz, y], dim=1))
        x_probs = nn.Sigmoid()(x_real)
        
        # Generative p(a|z,y,x)
        p_a_mu, p_a_logvar = self.aux_decoder(torch.cat([x,y,qz],dim=1))
        pa_dist = torch.distributions.Normal(p_a_mu, torch.exp(p_a_logvar.mul(0.5)))
        
        return x_probs, qa_dist, qz_dist, pa_dist, qa, qz, logits

    def loss(self, data, y, K=1, alpha=1.0, inference=inferences.VR, weight = 300., **kwargs):    
        priordist = torch.distributions.Normal(torch.zeros(data[0].size(0), self.zdim).to(self.device), torch.ones(data[0].size(0), self.zdim).to(self.device))  
        prior_z = [priordist] * K
        def inner_loss(data, y):
            qa_dist, qz_dist, pa_dist, qa, qz = [None] * K, [None] * K, [None] * K, [None] * K, [None] * K
            l_dist = [None] * K
            for k in range(K):
                x_probs, qa_dist[k], qz_dist[k], pa_dist[k], qa[k], qz[k], logits = self.forward(data, y)
                l_dist[k] = torch.distributions.Bernoulli(probs = x_probs)
            
            #compute loss function
            vr_loss = inference(qz, data.view(-1,784), l_dist, prior_z, qz_dist, K = K, alpha = alpha, reduce=False)
            #aux_loss = 0.0
            for k in range(K):
                if k is 0:
                    aux_loss = (pa_dist[k].log_prob(qa[k]) - qa_dist[k].log_prob(qa[k])).sum(dim=1,keepdim=True)
                else:
                    curr_aux_loss = (pa_dist[k].log_prob(qa[k]) - qa_dist[k].log_prob(qa[k])).sum(dim=1,keepdim=True)
                    aux_loss = torch.cat([aux_loss, curr_aux_loss],dim=1)
            aux_loss_lse = -LogSumExp(aux_loss, dim = 1)
            
            #print(vr_loss.cpu().data[0], aux_loss_lse.cpu().data[0])
            loss_total = vr_loss + aux_loss_lse
            return loss_total, logits
        
        if y is not None:
            vr_loss, logits = inner_loss(data, y)
            #eq 8, extended objective function in kingma
            
            c_dist = torch.distributions.OneHotCategorical(logits = logits)
            #print(y.size(), logits.size())
            c_loss = - weight * c_dist.log_prob(y)
            
            #print(vr_loss.size(), c_loss.size())
            tmp = vr_loss.view(-1) + c_loss
            #tmp = c_loss
            total_loss = (tmp).sum()
            
            secondary_loss = c_loss.sum()
        else:
            secondary_loss = None
            total_loss = 0.0
            for yy in range(10):
                #ycurrent = Variable(torch.ones(len(data))*yy)
                ycurrent = torch.zeros(data.size(0), 10).to(self.device)
                ycurrent[:,yy] = 1
                
                vr_loss, logits = inner_loss(data, ycurrent)
                y_dist = torch.distributions.OneHotCategorical(logits = logits)
                y_lprob = y_dist.log_prob(ycurrent)
                #consisten sizing
                total_loss += torch.exp(y_lprob) * (vr_loss.view(-1) - y_lprob)
        return total_loss.sum(), secondary_loss

    def calc_accuracy(self, data, y_one_hot, yvec):
        yvec = yvec.to(self.device)
        _, _, _, _, _, _, logits = self.forward(data, y_one_hot)
        _, pred = torch.max(logits, dim = 1)
        misclass = (pred.data.long() - yvec.long()).ne(int(0)).double().sum().cpu()
        return misclass/len(data)

class baseBADGM(baseADGM):
    def __init__(self, dim = 784, nclasses =10, zdim = 50, adim = 50, hidden = 500, activation = nn.ReLU):
        super(baseBADGM, self).__init__(dim, nclasses, zdim, adim, hidden, activation)

    def loss(self, data, y, **kwargs):
        main_loss, secondary_loss = super(baseBADGM, self).loss(data, y, **kwargs)
        prior_loss = compute_prior_loss(self) 

        return main_loss + prior_loss/len(data), secondary_loss

class ADGM:
    args = list()
    kwargs = {'dim': 784, 'hidden':500}
    base = baseADGM

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train

class BADGM(ADGM):
    def __init__(self):
        super(BADGM, self).__init__()
    base = baseBADGM