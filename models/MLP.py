import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary
class MLP(nn.Module):
    def __init__(self,
                 shape_in,
                 n_layers,
                 n_units,
                 n_inputs=1,
                 neurons_out=2,
                 bn=False,
                 act='tanh'):
        super(MLP, self).__init__()
        self.shape_in = shape_in
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_inputs = n_inputs
        self.neurons_out = neurons_out
        if not bn:
            self.bn = False
        else:
            raise NotImplementedError('Batchnorm not yet working, maybe layernorm?')
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
                layers.append(nn.BatchNorm1d(self.n_units))
        layers.append(self.act)
        layers.append(nn.Linear(self.n_units, self.neurons_out))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.repeat(*self.shape_in, 1).unsqueeze(-1)
        x = self.net(x)
        # return c_aif and c_tissue
        return x[..., 0], x[..., 1]

if __name__ == "__main__":
    model = MLP(np.ones((24, 24)),2,10,1)
    model(torch.rand(1))

    print(model)

