import torch
import torch.nn as nn
import numpy as np
from MLP import MLP

class PPINN(nn.Module):
    def __init__(self,
                 dummy_image,
                 n_layers,
                 n_units,
                 lr,
                 loss_weights=(1, 1, 1),
                 bn=False,
                 trainable_params='all',
                 n_inputs=1,
                 std_t=1):
        super(PPINN, self).__init__()
        self.lw_data, self. lw_pde, self.lw_bc = (0, 0, 0)
        self.optimizer = None
        self.var_list = None
        self.image_shape = dummy_image.shape
        self.indices = np.count_nonzero(dummy_image)
        self.std_t = std_t
        self.neurons_out = 1
        # initialize flow parameters
        low = 0
        high = 100
        self.flow_cbf = torch.nn.Parameter(torch.FloatTensor(self.image_shape[0] * self.image_shape[0], 1).uniform_(low, high))

        self.NN = MLP(
            dummy_image,
            n_layers,
            n_units,
            n_inputs=n_inputs,
            neurons_out=self.neurons_out,
            bn=bn,
            act='tanh'
        )

        self.set_lr(lr)
        self.set_loss_weights(loss_weights)
        self.set_trainable_params(trainable_params)

    def forward(self, t):
        params = self.get_ode_params()

        c_aif, c_tissue = self.NN(t)


    def set_loss_weights(self, loss_weights):
        self.lw_data, self. lw_pde, self.lw_bc = loss_weights

    def set_lr(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def set_trainable_params(self, trainable_params):
        if trainable_params == "all":
            self.var_list = self.parameters()
        else:
            raise NotImplementedError('Get to work and implement it!')

    def get_ode_pars(self):
        return self.flow_cbf

if __name__ == "__main__":
    testclass = PPINN(np.ones((24, 24)),
                      5,
                      5,5)






