import torch
from torch import nn
import torch.nn.functional as F
import math

class PreluLayer(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=None, init_std=None):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, hidden_dim))
        if init_std is None:
            torch.nn.init.kaiming_uniform_(self.W, a=0)
        else:
            torch.nn.init.normal_(self.W, std=init_std)

    def forward(self, xb, p, normalize_input):
        if p > 1:
            xh = torch.div(torch.pow(F.relu(xb @ self.W), p),
                           torch.pow(torch.linalg.norm(self.W, dim=0), p - 1))
            if normalize_input: 
                xh = torch.div(xh, torch.pow(torch.linalg.norm(xb, dim=1, keepdim=True), p - 1.0))
            else:
                pass
        else:
            xh = F.relu(xb @ self.W)

        return xh


# a network with pre-trained model as feature extractor followed by a probing network using prelu activations
class PreluProbing(nn.Module):
    def __init__(self, pretrained_model, prelu_prob, **prob_args):
        super().__init__()
        
        self.pretrained_model = pretrained_model
        pretrained_model.fc = nn.Sequential(
            nn.Identity()
        )
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.layernorm = nn.LayerNorm(fc_inputs)
        
        self.preluprob = prelu_prob(input_dim=fc_inputs, **prob_args)

    def forward(self, xb, p, normalize_input):
        xb = self.pretrained_model(xb)
        xb = self.layernorm(xb)
        xb = self.preluprob(xb, p, normalize_input)

        return x

# a single-hidden-layer prelu network
class ShallowPreluNet(nn.Module):
    def __init__(self, center=None, bias=None, input_dim=None, output_dim=None, hidden_dim=None, init_std=None):
        super().__init__()
        
        self.center = nn.Parameter(center, requires_grad=False) if center is not None else nn.Parameter(
            torch.zeros([1, input_dim]), requires_grad=False)
        
        self.prelulayer = PreluLayer(input_dim=input_dim, hidden_dim=hidden_dim, init_std=init_std)
        self.output_net = nn.Linear(hidden_dim, output_dim, bias=bias)
        if init_std is not None:
            nn.init.normal_(self.output_net.weight, std=init_std)

    def forward(self, xb, p, normalize_input=True, return_feature=False):
        xb = xb - self.center 
        xb = self.prelulayer(xb, p, normalize_input)
        if return_feature:
            return self.output_net(xb), xb
        else:
            return self.output_net(xb)


# fully-connected deep prelu network (w. or w.o. residual connection)
class DeepPreluNet(nn.Module):
    
    def __init__(self, center=None, bias=None, num_layers=10, skip_connect=False, input_dim=None, output_dim=None, hidden_dim=None, init_std=None, dropout=0.0):
        super().__init__()
        
        self.center = nn.Parameter(center, requires_grad=False) if center is not None else nn.Parameter(
            torch.zeros([1, input_dim]), requires_grad=False)

        self.input_net = PreluLayer(input_dim=input_dim, hidden_dim=hidden_dim, init_std=init_std)
        # self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.prelulayers = MultilayerPrelu(num_layers,
                                           skip_connect=skip_connect,
                                           hidden_dim=hidden_dim,
                                           init_std=init_std,
                                           dropout=dropout)

        self.output_net = nn.Linear(hidden_dim, output_dim, bias=bias)
        if init_std is not None:
            nn.init.normal_(self.out.weight, std=init_std)
            
    def forward(self, xb, p, normalize_input):
        xb = xb - self.center 
        xb = self.input_net(xb, p, normalize_input)
        # xb = self.input_norm(xb)
        xb = self.prelulayers(xb, p, normalize_input)
        xb = self.output_net(xb)

        return xb

class MultilayerPrelu(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([PreluBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, p, normalize_input):
        for l in self.layers:
            x = l(x, p, normalize_input)
        return x

class PreluBlock(nn.Module):

    def __init__(self, skip_connect=False, hidden_dim=None, init_std=None, dropout=0.0):
        super().__init__()
        
        self.skip_connect = skip_connect
        self.prelulayer = PreluLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, init_std=init_std)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        # self.norm = nn.Identity()
        
    def forward(self, x, p, normalize_input):
        x = self.prelulayer(x, p, normalize_input)
        if self.skip_connect:
            x = x + self.dropout(x)
        else:
            x = self.dropout(x)
        x = self.norm(x)
    
        return x



