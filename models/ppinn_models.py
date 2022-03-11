import torch
import torch.nn as nn
import numpy as np
from models.MLP import MLP
import time
from torch.utils.data import DataLoader
class PPINN(nn.Module):
    def __init__(self,
                 shape_in,
                 n_layers,
                 n_units,
                 lr,
                 loss_weights=(1, 1, 1),
                 bn=False,
                 trainable_params='all',
                 n_inputs=1,
                 std_t=1):
        super(PPINN, self).__init__()
        self.lw_data, self.lw_res, self.lw_bc = (0, 0, 0)
        self.optimizer = None
        self.var_list = None
        self.image_shape = shape_in
        self.std_t = std_t
        self.neurons_out = 1
        # initialize flow parameters
        low = 0
        high = 100
        self.flow_cbf = torch.nn.Parameter(torch.FloatTensor(self.image_shape[0] * self.image_shape[0], 1).uniform_(low, high))

        self.NN = MLP(
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

        residual = ...
        return c_aif, c_tissue, c_tissue_dt, params, residual

    def set_loss_weights(self, loss_weights):
        self.lw_data, self.lw_res, self.lw_bc = loss_weights

    def set_lr(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def set_trainable_params(self, trainable_params):
        if trainable_params == "all":
            self.var_list = self.parameters()
        else:
            raise NotImplementedError('Get to work and implement it!')

    def get_ode_pars(self):
        return self.flow_cbf

    def fit(self,
            data_time,
            data_aif,
            data_curves,
            data_collopoints,
            data_boundary,
            true_flow_params,
            batch_size,
            epochs):
        t0 = time.time()
        collopoints_dataloader = DataLoader(data_collopoints, batch_size=batch_size, shuffle=True)
        for ep in range(self.epochs + 1, self.epochs + epochs + 1):
            for batch_collopoints in collopoints_dataloader:
                batch_time = data_time
                batch_aif = data_aif
                batch_curves = data_curves
                batch_boundary = data_boundary
                batch_true_flow_params = true_flow_params
                self.optimize(batch_time,
                              batch_aif,
                              batch_curves,
                              batch_boundary,
                              batch_collopoints,
                              batch_true_flow_params)

    def optimize(self,
                 batch_time,
                 batch_aif,
                 batch_curves,
                 batch_boundary,
                 batch_collopoints,
                 batch_true_flow_params
                 ):
        self.optimizer.zero_grad()

        batch_time = batch_time.to(self.device)
        batch_aif = batch_aif.to(self.device)
        batch_curves = batch_curves.to(self.device)
        batch_boundary = batch_boundary.to(self.device)
        batch_collopoints = batch_collopoints.to(self.device)
        batch_true_flow_params = batch_true_flow_params.to(self.device)

        loss = torch.as_tensor(0.)

        if self.lw_data:
            # compute data loss
            output = self.forward(batch_time)
            loss += self.lw_data * self.__loss_data(batch_aif, batch_curves, output)
        if self.lw_res:

            # compute residual loss
            output = self.forward(batch_collopoints)
            loss += self.lw_res * self.__loss_residual(output)
        if self.lw_bc:
            # compute bc loss
            output = self.forward(batch_boundary)
            loss += self.lw_bc * self.__loss_bc(output)

    def __loss_data(self, aif, curves, output):
        c_aif, c_tissue, c_tissue_dt, _, _ = output
        loss_function = torch.nn.MSELoss()
        # solution loss
        loss_aif = loss_function(aif, c_aif)
        loss_tissue = loss_function(curves, c_tissue)
        # TODO maybe implement derivative loss here
        return loss_aif + loss_tissue

    def __loss_res(self, output):
        _, _, _, _, residual = output
        loss_r = torch.mean(torch.square(residual))
        return loss_r

    def __loss_bc(self, output):
        _, _, _, _, _ = output
        #TODO implement
        loss_bc = torch.mean(torch.square(residual))
        return loss_bc