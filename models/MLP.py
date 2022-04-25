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
            if self.bn:
                if self.aif:
                    layers.append(Rearrange('cbv n h w b t -> (cbv n h w b) t'))
                    layers.append(nn.BatchNorm1d(self.n_units))
                    layers.append(Rearrange('(cbv n h w b) t -> cbv n h w b t',
                                            cbv=self.shape_in[0],
                                            n=self.shape_in[1],
                                            h=self.shape_in[2],
                                            w=self.shape_in[3]))
                else:
                    layers.append(nn.BatchNorm1d(self.n_units))

        layers.append(self.act)
        if not self.aif:
            layers.append(nn.Linear(self.n_units,
                                self.neurons_out * self.shape_in[0] * self.shape_in[1] * self.shape_in[2] * self.shape_in[3]))
            layers.append(Rearrange('b (curves type cbv h w) -> type cbv h w b curves',
                                    curves=1,
                                    type=self.shape_in[0],
                                    cbv=self.shape_in[1],
                                    h=self.shape_in[2],
                                    w=self.shape_in[3]))
        else:
            layers.append(nn.Linear(self.n_units,
                                    self.neurons_out))
        return nn.Sequential(*layers)

    def forward(self, x):

        # x = x.repeat(*self.shape_in, 1).unsqueeze(-1)
        x = self.net(x)
        # if not self.aif:
        #     x = x.view(*self.shape_in, -1)
        # return c_aif and c_tissue
        return x[...,0]
