import torch
import torch.nn as nn
import numpy as np
from models.MLP import MLP
import time
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
class PPINN(nn.Module):
    def __init__(self,
                 shape_in,
                 n_layers,
                 n_units,
                 lr,
                 loss_weights=(1, 1, 0),
                 bn=False,
                 trainable_params='all',
                 n_inputs=1,
                 std_t=1):
        super(PPINN, self).__init__()
        self.device = 'cuda'
        self.lw_data, self.lw_res, self.lw_bc = (0, 0, 0)
        self.optimizer = None
        self.interpolator = None
        self.var_list = None
        self.shape_in = shape_in
        self.std_t = std_t
        self.neurons_out = 2
        # initialize flow parameters
        low = 0
        high = 100
        self.flow_cbf = torch.nn.Parameter(torch.FloatTensor(*self.shape_in, 1).uniform_(low, high))
        self.fixed_mtt = torch.tensor(-1.4776831518357147)
        self.fixed_mtt.requires_grad = False
        self.fixed_mtt.to(self.device)
        self.NN = MLP(
            self.shape_in,
            n_layers,
            n_units,
            n_inputs=n_inputs,
            neurons_out=self.neurons_out,
            bn=bn,
            act='tanh'
        )
        self.current_iteration = 0
        self.set_lr(lr)
        self.set_loss_weights(loss_weights)
        self.set_trainable_params(trainable_params)
        self.set_device(self.device)
        self.float()

    def forward(self, t):
        # Get ODE params
        params = self.get_ode_params()
        # Get NN output
        c_aif, c_tissue = self.NN(t)
        # Get time-derivative of tissue curve
        c_tissue_dt = (1 / self.std_t) * self.__fwd_gradients(c_tissue, t)
        # if self.current_iteration %10 == 0:
        #     plt.plot(c_tissue_dt.cpu().squeeze().detach().numpy())
        #     plt.show()
        # coordinates = (t,)

        # print((1 / self.std_t) * torch.gradient(c_tissue.squeeze(), spacing=coordinates)[0])
        # Get NN output at MTT time
        t.requires_grad = False
        # print(t-self.fixed_mtt)
        c_aif_b, _ = self.NN(t-3.4/self.std_t)
        # c_aif_b = None
        # print('sizes')
        # print(c_tissue_dt)
        # print((c_aif - c_aif_b))
        # Define residual
        residual = c_tissue_dt - 10**-4*params * (c_aif - c_aif_b)
        # residual = None

        # print(residual.shape)
        return c_aif, c_aif_b, c_tissue, c_tissue_dt, params, residual

    def set_loss_weights(self, loss_weights):
        loss_weights = torch.tensor(loss_weights)
        self.lw_data, self.lw_res, self.lw_bc = loss_weights
        self.lw_data.to(self.device)
        self.lw_res.to(self.device)
        self.lw_bc.to(self.device)

    def set_lr(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def set_device(self, device):
        self.to(device)

    def set_trainable_params(self, trainable_params):
        if trainable_params == "all":
            self.var_list = self.parameters()
        else:
            raise NotImplementedError('Get to work and implement it!')

    def get_ode_params(self):
        return self.flow_cbf

    def define_interpolator(self, time, aif, mode='quadratic'):
        self.interpolator = interp1d(time, aif,
                                     kind=mode,
                                     bounds_error=False,
                                     fill_value=(aif[0], aif[-1]),
                                     assume_sorted=False)

    def fit(self,
            data_time,
            data_aif,
            data_curves,
            data_collopoints,
            data_boundary,
            batch_size,
            epochs):

        t0 = time.time()
        collopoints_dataloader = DataLoader(data_collopoints, batch_size=batch_size, shuffle=True)
        for ep in range(self.current_iteration + 1, self.current_iteration + epochs + 1):
            for batch_collopoints in collopoints_dataloader:
                batch_time = data_time
                batch_aif = data_aif
                batch_curves = data_curves
                batch_boundary = data_boundary
                self.optimize(batch_time,
                              batch_aif,
                              batch_curves,
                              batch_boundary,
                              batch_collopoints, ep)
                # optimized_parameters = self.get_ode_params()
                # plt.imshow(optimized_parameters[2,4,:,:,0].cpu().detach().numpy())
                # plt.show()


    def optimize(self,
                 batch_time,
                 batch_aif,
                 batch_curves,
                 batch_boundary,
                 batch_collopoints,
                 epoch):
        self.train()
        self.optimizer.zero_grad()

        batch_time = batch_time.to(self.device)
        batch_aif = batch_aif.to(self.device)
        batch_curves = batch_curves.to(self.device)
        batch_boundary = batch_boundary.to(self.device)
        batch_collopoints = batch_collopoints.to(self.device)
        batch_time.requires_grad = True
        batch_collopoints.requires_grad = True
        batch_boundary.requires_grad = True
        loss = torch.as_tensor(0.).to(self.device)

        if self.lw_data:
            # compute data loss
            output = self.forward(batch_time)
            loss += self.lw_data * self.__loss_data(batch_aif, batch_curves, output)

            c_aif, c_aif_b, c_tissue, c_tissue_dt, params, residual = output

            # if epoch%30 == 0:
            #     plt.plot(batch_time.cpu().numpy(),c_tissue[0,0,0,0].cpu().detach().numpy(), c='r')
            #     plt.plot(batch_time.cpu().numpy(), batch_curves[0,0,0,0].cpu().detach().numpy(), c='g')
            #     plt.text(0,0.17,epoch)
            #     # plt.plot(batch_time.cpu().numpy(),batch_aif.cpu().numpy(), c='k')
            #     plt.show()


        if self.lw_res:
            # compute residual loss
            output = self.forward(batch_collopoints)
            c_aif, c_aif_b, c_tissue, c_tissue_dt, params, residual = output
            loss += self.lw_res * self.__loss_residual(output)
        if self.lw_bc:
            # compute bc loss
            output = self.forward(batch_boundary)
            loss += self.lw_bc * self.__loss_bc(output)
        # todo implepent backwars pass
        if np.isnan(float(loss.item())):
            raise ValueError('Loss is nan during training...')


        # optimizer
        loss.backward()
        self.optimizer.step()
        self.current_iteration += 1

    def __loss_data(self, aif, curves, output):
        c_aif, _, c_tissue, _, _, _ = output
        # reshape the ground truth
        aif = aif.expand(*c_aif.shape)
        # solution loss
        loss_aif = F.mse_loss(aif, c_aif)
        loss_tissue = F.mse_loss(curves, c_tissue)
        # TODO maybe implement derivative loss here
        return loss_aif + loss_tissue

    def __loss_interpolation(self, aif, curves, output):
        # TODO implement loss that uses C_aif(t-MTT) estimation and compares to interpolated version of AIF
        pass

    def __loss_residual(self, output):
        _, _, _, _, _, residual = output
        loss_r = 1 * torch.mean(torch.square(residual))
        print(self.get_ode_params())
        return loss_r

    def __loss_bc(self, output):
        _, _, _, _, _ = output
        #TODO implement
        loss_bc = 0
        return loss_bc

    def __fwd_gradients(self, ys, xs):
        v = torch.ones_like(ys)
        v.requires_grad = True
        g = torch.autograd.grad(
            outputs=[ys],
            inputs=xs,
            grad_outputs=[v],
            create_graph=True,
        )[0]

        w = torch.ones_like(g)
        w.requires_grad = True
        out = torch.autograd.grad(
            outputs=[g],
            inputs=v,
            grad_outputs=[w],
            create_graph=True,
        )[0]
        return out