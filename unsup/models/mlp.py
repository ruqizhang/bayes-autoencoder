import torch
from torch import nn

class Linear2(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear2, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        self.layer_1, self.layer_2 = nn.Linear(self.in_features, self.out_features), nn.Linear(self.in_features, self.out_features) 
    def forward(self, x):
        x = x.view(-1, self.in_features)
        return [self.layer_1(x), self.layer_2(x)]
    
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_layer = nn.Linear, activation = nn.ReLU):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation
        self.out_layer = out_layer(hidden_features, out_features)
        
        self.model = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                self.activation(),
                self.out_layer)
        
    def forward(self, x):
        x = x.view(-1, self.in_features)
        return self.model(x)