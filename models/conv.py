import torch.nn as nn
import torch
import numpy as np
from einops.layers.torch import Rearrange
from torchsummary import summary
class PreConv(nn.Module):
    def __init__(self):
        super(PreConv, self).__init__()
        self.n_layers = 1
        self.net = self.__make_net()

    def __make_net(self):
        layers = [Rearrange('cbv (c n) h w t -> (cbv n t) c h w', c=1)]
        for i in range(self.n_layers):
            layers.append(nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, padding='same', bias=False))
            # layers.append(nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, padding='same', bias=False))
        layers.append(Rearrange('(cbv n t) c h w -> (c cbv) n h w t', cbv=1, n=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_conv = self.net(x)
        x_conv = torch.mean(x_conv, dim=0).unsqueeze(0)
        return x_conv
