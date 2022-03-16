import torch
import torch.nn as nn
import numpy as np
from models.MLP import MLP
import time
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
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
        # self.hemo_scaling = torch.as_tensor(((100*0.55)/(1.05*0.75))).to(self.device)
        # initialize flow parameters
        low = 0
        high = 100 / 69.84
        self.flow_cbf = torch.nn.Parameter(torch.FloatTensor(*self.shape_in, 1).uniform_(low, high))
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
        # Get NN output at MTT time
        t.requires_grad = False
        c_aif_b, _ = self.NN(t-4./self.std_t)
        # print('sizes')
        # print(c_tissue_dt)
        # print((c_aif - c_aif_b))
        # Define residual
        residual = c_tissue_dt - params * (c_aif - c_aif_b)
        # print(residual)
        # residual = None
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

    def get_cbf(self, seconds: object = True):
        density = 1.05
        constant = (100/density) * 0.55 / 0.75
        constant = torch.as_tensor(constant).to(self.device)
        f_s = self.get_ode_params()
        if seconds:
            return constant * f_s
        else:
            return constant * f_s * 60


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
            gt,
            batch_size,
            epochs):

        t0 = time.time()
        collopoints_dataloader = DataLoader(data_collopoints, batch_size=batch_size, shuffle=True)
        for ep in tqdm(range(self.current_iteration + 1, self.current_iteration + epochs + 1)):
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
            if ep%300 == 0 or ep ==1:
                # print(self.get_cbf(seconds=False))
                self.plot_params(0,0,gt,ep)
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
        if self.lw_res:
            # compute residual loss
            output = self.forward(batch_collopoints)
            loss += self.lw_res * self.__loss_residual(output)
        if self.lw_bc:
            # compute bc loss
            output = self.forward(batch_boundary)
            loss += self.lw_bc * self.__loss_bc(output)

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
        # if epoch%5000==0:
        #     plt.title('residuals squared')
        #     plt.imshow(torch.square(residual)[0,0,:,:,0].cpu().detach().numpy())
        #     plt.show()
        loss_r = torch.mean(torch.square(residual))
        # print(self.get_ode_params())
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

    def plot_params(self, i, j, perfusion_values, epoch):
        params = self.get_cbf(seconds=False)
        # print(self.get_ode_params())
        for par in range(params.shape[-1]):
            print(params[i,j,:,:,par])
            fig, ax = plt.subplots(2, 1)
            ax[0].set_title('Epoch: {}'.format(epoch))
            im = ax[0].imshow(params[i,j,:,:,par].cpu().detach().numpy(), vmin=0, vmax=90)
            im = ax[1].imshow(perfusion_values[i,j,:,:,par].cpu().detach().numpy(), vmin=0, vmax=90)
            for x in ax:
                x.set_axis_off()
            plt.tight_layout()
            fig.colorbar(im, ax=ax.ravel().tolist(),location="bottom")
            plt.show()
