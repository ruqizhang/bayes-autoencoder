import torch
from torch import nn
from torchvision.utils import save_image
from mlp import MLP, Linear2
import torch.distributions

class ADGM(nn.Module):
    #heavily based on https://github.com/wohlert/semi-supervised-pytorch
    def __init__(self, xdim = 784, nclasses =10, zdim = 50, adim = 50, hidden = 500, activation = nn.ReLU):
        super(ADGM, self).__init__()
        self.xdim, self.nclasses, self.zdim, self.adim, self.hidden = xdim, nclasses, zdim, adim, hidden
        
        self.aux_encoder = MLP(xdim, hidden, adim, out_layer = Linear2, activation = activation)
        self.aux_decoder = MLP(xdim + zdim + nclasses, hidden, adim, out_layer = Linear2, activation = activation)
        
        self.encoder_y_real = MLP(xdim + adim, hidden, nclasses, activation = activation)
        self.encoder = MLP(adim + nclasses + xdim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(nclasses + zdim, hidden, xdim)
        
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