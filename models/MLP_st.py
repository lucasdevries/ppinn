import torch.nn as nn
import torch
import numpy as np
from einops.layers.torch import Rearrange
from torchsummary import summary
class MLP(nn.Module):
    def __init__(self,
                 shape_in,
                 aif,
                 n_layers,
                 n_units,
                 n_inputs=1,
                 neurons_out=1,
                 bn=False,
                 act='tanh'):
        super(MLP, self).__init__()
        self.shape_in = shape_in
        self.aif = aif
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.neurons_out = neurons_out
        if not bn:
            self.bn = False
        else:
            raise NotImplementedError('Batchnorm not yet working, maybe layernorm?')
            # self.bn = True
        if act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise NotImplementedError("There is no other activation implemented.")

        self.net = self.__make_net()

    def __make_net(self):
        layers = [nn.Linear(self.n_inputs, self.n_units)]
        for i in range(self.n_layers):
            layers.append(self.act)
            layers.append(nn.Linear(self.n_units, self.n_units))
        layers.append(self.act)
        layers.append(nn.Linear(self.n_units,
                                self.neurons_out))
        return nn.Sequential(*layers)

    def forward(self, t, xy):
        if not self.aif:
            txy = torch.concat([t, xy], dim=-1)

            out = self.net(txy)
            return out
        else:
            t = self.net(t)
            return t[..., 0]

class MLP_ODE(nn.Module):
    def __init__(self,
                 n_layers,
                 n_units,
                 n_inputs=1,
                 neurons_out=1,
                 bn=False,
                 act='tanh'):
        super(MLP_ODE, self).__init__()
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.neurons_out = neurons_out
        if not bn:
            self.bn = False
        else:
            raise NotImplementedError('Batchnorm not yet working, maybe layernorm?')
            # self.bn = True
        if act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise NotImplementedError("There is no other activation implemented.")

        self.net = self.__make_net()

    def __make_net(self):
        layers = [nn.Linear(self.n_inputs, self.n_units)]
        for i in range(self.n_layers):
            layers.append(self.act)
            layers.append(nn.Linear(self.n_units, self.n_units))
        layers.append(self.act)
        layers.append(nn.Linear(self.n_units,
                                self.neurons_out))
        return nn.Sequential(*layers)

    def forward(self, xy):
        out = self.net(xy)
        return out