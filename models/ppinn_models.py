import torch
import torch.nn as nn
import numpy as np
from models.MLP import MLP
import time
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops.einops import repeat
from tqdm import tqdm
class PPINN(nn.Module):
    def __init__(self,
                 shape_in,
                 n_layers,
                 n_units,
                 lr,
                 perfusion_values,
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
        self.neurons_out = 1

        # if using GT mtt and cbf
        # self.mtts = perfusion_values[..., 2] * 60
        # self.mtts = self.mtts.view(*self.mtts.shape, 1,1 )
        # self.mtts = self.mtts.to(self.device)
        #
        # self.truecbf = perfusion_values[..., -1]
        # self.truecbf = self.truecbf.view(*self.truecbf.shape, 1)
        # self.truecbf = self.truecbf.to(self.device)

        # density = 1.05
        # constant = (100/density) * 0.55 / 0.75
        # self.truecbf = self.truecbf / constant

        # if using set delay values
        self.delays = perfusion_values[..., 1]
        self.delays = self.delays.view(*self.delays.shape, 1,1 )
        # self.delays = torch.zeros_like(self.delays)
        self.delays = self.delays.to(self.device)
        # initialize flow parameters
        low = 0
        high = 100 / (69.84*60)
        # self.flow_cbf = torch.nn.Parameter(torch.FloatTensor(*self.shape_in, 1).uniform_(low, high))
        # plt.imshow(torch.nn.Parameter(flow[0,0,:,:,0]), cmap='jet')
        # plt.show()
        self.flow_cbf = torch.nn.Parameter(torch.rand(*self.shape_in, 1)*high)
        self.flow_mtt = torch.nn.Parameter(torch.rand(*self.shape_in, 1, 1))
        self.flow_t_delay = torch.nn.Parameter(torch.rand(*self.shape_in, 1, 1))

        nn.init.uniform_(self.flow_cbf, low, high)
        nn.init.uniform_(self.flow_mtt)
        self.NN_tissue = MLP(
            self.shape_in,
            False,
            n_layers,
            n_units,
            n_inputs=n_inputs,
            neurons_out=1,
            bn=bn,
            act='tanh'
        )
        self.NN_aif = MLP(
            self.shape_in,
            True,
            n_layers,
            n_units,
            n_inputs=n_inputs,
            neurons_out=1,
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
        length = t.shape[0]
        # Get ODE params
        params = self.get_ode_params()
        t = t.unsqueeze(-1)
        # Get NN output: a tissue curve for each voxel
        c_tissue = self.NN_tissue(t)
        # Get time-derivative of tissue curve
        c_tissue_dt = (1 / self.std_t) * self.__fwd_gradients(c_tissue, t)


        # Get AIF NN output:
        t = t.view(1,1,1,1,length, 1)
        t = t.expand(*self.shape_in, length, 1)


        # Get NN output at MTT time
        t = t.detach()
        delay_s = self.delays.expand(*self.shape_in,length,1)
        delay_s.requires_grad = False
        t.requires_grad = False

        c_aif = self.NN_aif(t-delay_s/self.std_t)

        input = t - delay_s/self.std_t - 24*params[1]/self.std_t
        c_aif_b = self.NN_aif(input)
        # Define residual
        residual = c_tissue_dt - params[0] * (c_aif - c_aif_b)



        # print(residual)
        # if using set cbf and mtt
        # input = t - 3*params[2]/self.std_t - self.mtts/self.std_t
        #
        # c_aif_b = self.NN_aif(input)
        # residual = c_tissue_dt - self.truecbf * (c_aif - c_aif_b)

        return c_aif, c_aif_b, c_tissue, c_tissue_dt, params, residual
    def forward_allfree(self, t):
        length = t.shape[0]
        # Get ODE params
        params = self.get_ode_params()
        t = t.unsqueeze(-1)
        # Get NN output: a tissue curve for each voxel
        c_tissue = self.NN_tissue(t)
        # Get time-derivative of tissue curve
        c_tissue_dt = (1 / self.std_t) * self.__fwd_gradients(c_tissue, t)


        # Get AIF NN output:
        t = t.view(1,1,1,1,length, 1)
        t = t.expand(*self.shape_in, length, 1)


        # Get NN output at MTT time
        t = t.detach()

        t.requires_grad = False

        c_aif = self.NN_aif(t-3*params[2]/self.std_t)

        input = t - 3*params[2]/self.std_t - 24*params[1]/self.std_t
        c_aif_b = self.NN_aif(input)
        # Define residual
        residual = c_tissue_dt - params[0] * (c_aif - c_aif_b)
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
        return [self.flow_cbf, self.flow_mtt, self.flow_t_delay]

    def get_mtt(self, seconds=True):
        _, mtt_par, _ = self.get_ode_params()
        if seconds:
            return 24*mtt_par.squeeze(-1)
        else:
            return 24*mtt_par.squeeze(-1)/60

    def get_cbf(self, seconds=True):
        density = 1.05
        constant = (100/density) * 0.55 / 0.75
        constant = torch.as_tensor(constant).to(self.device)
        f_s, _, _ = self.get_ode_params()
        if seconds:
            return constant * f_s
        else:
            return constant * f_s * 60

    def get_delay(self, seconds=True):
        _, _, delay = self.get_ode_params()
        if seconds:
            return 3*delay.squeeze(-1)
        else:
            return 3*delay.squeeze(-1)/60
        # return self.delays.squeeze(-1).squeeze(-1)


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
            if ep%500 == 0:
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
            output = self.forward_allfree(batch_collopoints)
            loss += self.lw_res * self.__loss_residual(output)

        if self.lw_bc:
            # compute bc loss
            output = self.forward(batch_boundary)
            loss += self.lw_bc * self.__loss_bc(output)

        if np.isnan(float(loss.item())):
            raise ValueError('Loss is nan during training...')

        # optimizer
        loss.backward()
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         if name == 'flow_t_delay':
        #             param.grad *= 10000000

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
        loss_r = torch.mean(torch.square(residual))
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
        cbf = self.get_cbf(seconds=False).squeeze(-1)
        mtt = self.get_mtt(seconds=True).squeeze(-1)
        mtt_min = self.get_mtt(seconds=False).squeeze(-1)
        delay = self.get_delay(seconds=True).squeeze(-1)

        cbv = cbf * mtt_min

        # print(torch.min(mtt),torch.min(cbf))
        # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
        gt_cbv = perfusion_values[..., 0]
        gt_delay = perfusion_values[..., 1]
        gt_mtt = perfusion_values[..., 2]*60
        gt_cbf = perfusion_values[..., 3]

        [cbf, mtt, cbv, gt_cbf, gt_mtt, gt_cbv, delay] = [x.detach().cpu().numpy() for x in [cbf, mtt, cbv, gt_cbf, gt_mtt, gt_cbv, delay]]

        i, j = 0, 0

        fig, ax = plt.subplots(1, 4)
        ax[0].hist(cbf[i,j].flatten(), bins=100)
        ax[0].set_title('cbf')
        ax[1].hist(mtt[i,j].flatten(), bins=100)
        ax[1].set_title('mtt')
        ax[2].hist(cbv[i,j].flatten(), bins=100)
        ax[2].set_title('cbv')
        ax[3].hist(delay[i,j].flatten(), bins=100, range=(0,3))
        ax[3].set_title('delay')
        plt.show()
        # fig, ax = plt.subplots(1, 4)
        # ax[0].hist(cbf[i,j].flatten(), bins=100, range=(0,150))
        # ax[0].set_title('cbf')
        # ax[1].hist(mtt[i,j].flatten(), bins=100, range=(0,30))
        # ax[1].set_title('mtt')
        # ax[2].hist(cbv[i,j].flatten(), bins=100, range=(0,10))
        # ax[2].set_title('cbv')
        # ax[3].hist(delay[i,j].flatten(), bins=100, range=(0,3))
        # ax[3].set_title('delay')
        # plt.show()

        fig, ax = plt.subplots(2, 4)

        ax[0,0].set_title('CBF (ml/100g/min)')
        ax[0,0].imshow(cbf[i,j], vmin=0, vmax=150, cmap='jet')
        im = ax[1,0].imshow(gt_cbf[i,j], vmin=0, vmax=150, cmap='jet')
        fig.colorbar(im, ax=ax[1,0], location="bottom")

        ax[0,1].set_title('MTT (s)')
        ax[0,1].imshow(mtt[i,j], vmin=0, vmax=25, cmap='jet')
        im = ax[1,1].imshow(gt_mtt[i,j], vmin=0, vmax=25, cmap='jet')
        fig.colorbar(im, ax=ax[1,1], location="bottom")

        ax[0,2].set_title('CBV (ml/100g)')
        ax[0,2].imshow(cbv[i,j], vmin=0, vmax=10, cmap='jet')
        im = ax[1,2].imshow(gt_cbv[i,j], vmin=0, vmax=10, cmap='jet')
        fig.colorbar(im, ax=ax[1,2], location="bottom")

        ax[0,3].set_title('Delay (s)')
        im = ax[0,3].imshow(delay[i,j], vmin=0, vmax=3, cmap='jet')
        im = ax[1,3].imshow(gt_delay[i,j], vmin=0, vmax=3, cmap='jet')

        fig.colorbar(im, ax=ax[1,3], location="bottom")

        for x in ax.flatten():
            x.set_axis_off()
        fig.suptitle('Parameter estimation epoch: {}'.format(epoch))
        plt.tight_layout()
        plt.show()
