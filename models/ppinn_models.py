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
                 std_t=1,
                 delay='fixed'):
        super(PPINN, self).__init__()
        self.device = 'cuda'
        self.lw_data, self.lw_res, self.lw_bc = (0, 0, 0)
        self.optimizer = None
        self.interpolator = None
        self.var_list = None
        self.shape_in = shape_in
        self.std_t = std_t
        self.neurons_out = 1
        self.perfusion_values = perfusion_values
        # initialize flow parameters
        low = 0
        high = 100 / (69.84*60)
        self.flow_cbf = torch.nn.Parameter(torch.rand(*self.shape_in, 1)*high)
        self.flow_mtt = torch.nn.Parameter(torch.rand(*self.shape_in, 1, 1))
        self.delay_type = delay
        self.set_delay_parameter()

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

    def get_delay_bolus_arrival_time(self,t):
        find_max_batch = torch.arange(torch.min(t).item(), torch.max(t).item(), step=0.01).unsqueeze(-1).to(self.device)
        aif_estimation = self.NN_aif(find_max_batch)
        aif_dt = torch.gradient(aif_estimation)[0]
        aif_dtdt = torch.gradient(aif_dt)[0]
        top_2_maximums = torch.topk(aif_dtdt, k=2)
        index_of_bolus_arrival = torch.min(top_2_maximums.indices).item()
        bat_aif = find_max_batch[index_of_bolus_arrival]
        # plt.plot(aif_estimation.cpu().detach().numpy(), label='curve')
        # plt.plot(10 * aif_dt.cpu().detach().numpy(), label='first derivative')
        # plt.plot(10 * aif_dtdt.cpu().detach().numpy(), label='second derivative')
        # plt.scatter(index_of_bolus_arrival, aif_estimation.cpu().detach().numpy()[index_of_bolus_arrival],
        #             label='Bolus arrival', c='k')
        # plt.legend()
        # plt.show()
        find_max_batch.requires_grad = True
        tac_estimation = self.NN_tissue(find_max_batch)
        tac_dt = self.__fwd_gradients(tac_estimation, find_max_batch)
        tac_dtdt = self.__fwd_gradients(tac_dt, find_max_batch)
        # tac_dtdtdt = self.__fwd_gradients(tac_dtdt, find_max_batch)
        # for i in range(16, 17):
        #     for j in range(16, 224-16, 32):
        #         plt.plot(find_max_batch.detach().cpu().numpy()*self.std_t.numpy(),
        #                  tac_estimation[0,0,i,j].detach().cpu().numpy())
        # plt.plot(find_max_batch.detach().cpu().numpy() * self.std_t.numpy(),
        #          aif_estimation.detach().cpu().numpy())
        # plt.ylim(0.1, 0.4)
        # plt.show()
        # for i in range(10):
        #     for j in range(10):
        #         plt.plot(tac_dt[0,0,i,j].detach().cpu().numpy())
        # plt.show()
        # for i in range(10):
        #     for j in range(10):
        #         plt.plot(tac_dtdt[0,0,i,j].detach().cpu().numpy())
        # plt.show()
        # for i in range(10):
        #     for j in range(10):
        #         plt.plot(tac_dtdtdt[0,0,i,j].detach().cpu().numpy())
        # plt.show()

        top_2_maximums_tac = torch.topk(tac_dtdt, k=2, dim=-1)
        top_2_maximum_tac_indices = top_2_maximums_tac.indices.data
        index_of_bolus_arrival_tac = torch.min(top_2_maximum_tac_indices, dim=-1).values
        find_max_batch = find_max_batch.expand(*self.shape_in, len(find_max_batch), 1)
        index_of_bolus_arrival_tac = index_of_bolus_arrival_tac.unsqueeze(-1).unsqueeze(-1).long()
        bat_tac = torch.gather(find_max_batch, dim=-2, index=index_of_bolus_arrival_tac)

        delay = bat_tac - bat_aif
        delay *= self.std_t
        if self.current_iteration == 1999:
            plt.plot(aif_estimation.cpu().detach().numpy(), c='k')
            plt.scatter(index_of_bolus_arrival, 0.5)

            for i in range(16, 17):
                for j in range(16,208, 32):
                    plt.plot(tac_estimation.cpu().detach().numpy()[0,0,i,j])
                    plt.scatter(index_of_bolus_arrival_tac.cpu().detach().numpy()[0,0,i,j], 0.5)
            plt.show()

        return delay


        # return bolus_arrival_aif, bolus_arrival_tissue
    def get_delay_between_peaks(self,t):
        find_max_batch = torch.arange(torch.min(t).item(), torch.max(t).item(), step=0.01).unsqueeze(-1).to(self.device)
        # get maximum time of tac curve
        tac_estimation = self.NN_tissue(find_max_batch)
        largest_tac = find_max_batch[torch.topk(tac_estimation, k=1, dim=-1).indices].squeeze(-1)
        # get maximum time of aif curve
        aif_estimation = self.NN_aif(find_max_batch)
        largest_aif = find_max_batch[torch.topk(aif_estimation, k=1, dim=-1).indices]
        largest_aif = largest_aif.expand(*self.shape_in, 1)

        delay = largest_tac - largest_aif
        delay *= self.std_t
        return delay.unsqueeze(-1)

    def forward_NNs(self, t):
        t = t.unsqueeze(-1)
        # Get NN output: a tissue curve for each voxel
        c_tissue = self.NN_tissue(t)
        c_aif = self.NN_aif(t)

        #     # for i in range(16, 208, 32):
        #     #     for j in range(16, 208, 32):
        #     #         get_bolus_arrival_time(c_tissue.cpu().detach()[0, 0, i, j, :])
        #     for i in range(0, 224):
        #         for j in range(0, 2):
        #             get_bolus_arrival_time(c_tissue.cpu().detach()[0, 0, i, j, :])
        #     plt.show()
        #
        #     get_bolus_arrival_time(c_aif.cpu().detach())
        #     plt.show()
        #     print('hoi')
        return c_aif, c_tissue

    def forward_complete(self, t):
        t = t.unsqueeze(-1)
        steps = t.shape[0]
        # Get NN output: a tissue curve for each voxel
        c_tissue = self.NN_tissue(t)
        c_aif = self.NN_aif(t)
        # Get time-derivative of tissue curve
        c_tissue_dt = (1 / self.std_t) * self.__fwd_gradients(c_tissue, t)
        # Get ODE params
        cbf, mtt = self.get_ode_params()

        if self.delay_type == 'learned':
            delay = self.get_delay()
        elif self.delay_type == 'calculated_peak':
            # get delay between peaks, not yet corrected for mtt
            delay = self.get_delay_between_peaks(t)
            delay -= 24 * mtt / 2
            self.flow_t_delay = delay.to(self.device) / 3
            delay = self.get_delay()
        elif self.delay_type == 'calculated_bat':
            delay = self.get_delay_bolus_arrival_time(t)
            self.flow_t_delay = delay.to(self.device) / 3
            delay = self.get_delay()
        elif self.delay_type == 'fixed':
            delay = self.get_delay()
            delay.requires_grad = False
        else:
            raise NotImplementedError('Delay type not implemented...')

        # Get AIF NN output:
        t = t.view(1,1,1,1,steps, 1)
        t = t.expand(*self.shape_in, steps, 1)
        t = t.detach()

        t.requires_grad = False
        delay = delay.expand(*self.shape_in, steps, 1)
        c_aif_a = self.NN_aif(t - delay/self.std_t)
        c_aif_b = self.NN_aif(t - delay/self.std_t - 24*mtt/self.std_t)

        residual = c_tissue_dt - cbf * (c_aif_a - c_aif_b)

        return c_aif, c_tissue, residual

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
    def set_delay_parameter(self):
        if self.delay_type == 'learned':
            self.flow_t_delay = torch.nn.Parameter(torch.rand(*self.shape_in, 1, 1))
        elif self.delay_type == 'calculated_peak':
            self.flow_t_delay = torch.rand(*self.shape_in, 1, 1).to(self.device)
        elif self.delay_type == 'calculated_bat':
            self.flow_t_delay = torch.rand(*self.shape_in, 1, 1).to(self.device)
        elif self.delay_type == 'fixed':
            self.flow_t_delay = self.perfusion_values[..., 1] / 3
            self.flow_t_delay = self.flow_t_delay.view(*self.flow_t_delay.shape, 1,1)
            self.flow_t_delay = self.flow_t_delay.to(self.device)
        else:
            raise NotImplementedError('Delay type not implemented...')
    def get_ode_params(self):
        return [self.flow_cbf, self.flow_mtt]

    def get_delay(self, seconds=True):
        return 3 * self.flow_t_delay

    def get_mtt(self, seconds=True):
        _, mtt_par = self.get_ode_params()
        if seconds:
            return 24*mtt_par.squeeze(-1)
        else:
            return 24*mtt_par.squeeze(-1)/60

    def get_cbf(self, seconds=True):
        density = 1.05
        constant = (100/density) * 0.55 / 0.75
        constant = torch.as_tensor(constant).to(self.device)
        f_s, _ = self.get_ode_params()
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
                              batch_collopoints)
            if ep%500 == 0:
                # print(self.get_cbf(seconds=False))
                self.plot_params(0,0,gt,ep)
            self.current_iteration += 1

    def optimize(self,
                 batch_time,
                 batch_aif,
                 batch_curves,
                 batch_boundary,
                 batch_collopoints
                 ):
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

        if self.current_iteration < 0:
            if self.lw_data:
                # compute data loss
                c_aif, c_tissue = self.forward_NNs(batch_time)
                loss += self.lw_data * self.__loss_data(batch_aif, batch_curves, c_aif, c_tissue)
        else:
            if self.lw_data:
                # compute data loss
                c_aif, c_tissue = self.forward_NNs(batch_time)
                loss += self.lw_data * self.__loss_data(batch_aif, batch_curves, c_aif, c_tissue)
            if self.lw_res:
                # compute residual loss
                c_aif, c_tissue, residual = self.forward_complete(batch_collopoints)
                loss += self.lw_res * self.__loss_residual(residual)
            if self.lw_bc:
                # compute bc loss
                output = self.forward(batch_boundary)
                loss += self.lw_bc * self.__loss_bc(output)

        if np.isnan(float(loss.item())):
            raise ValueError('Loss is nan during training...')

        loss.backward()
        self.optimizer.step()

    def __loss_data(self, aif, curves, c_aif, c_tissue):
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

    def __loss_residual(self, residual):
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

        # fig, ax = plt.subplots(1, 4)
        # ax[0].hist(cbf[i,j].flatten(), bins=100)
        # ax[0].set_title('cbf')
        # ax[1].hist(mtt[i,j].flatten(), bins=100)
        # ax[1].set_title('mtt')
        # ax[2].hist(cbv[i,j].flatten(), bins=100)
        # ax[2].set_title('cbv')
        # ax[3].hist(delay[i,j].flatten(), bins=100, range=(0,3))
        # ax[3].set_title('delay')
        # plt.show()
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
        ax[0,0].imshow(cbf[i,j], vmin=0, vmax=90, cmap='jet')
        im = ax[1,0].imshow(gt_cbf[i,j], vmin=0, vmax=90, cmap='jet')
        fig.colorbar(im, ax=ax[1,0], location="bottom")

        ax[0,1].set_title('MTT (s)')
        ax[0,1].imshow(mtt[i,j], vmin=0, vmax=25, cmap='jet')
        im = ax[1,1].imshow(gt_mtt[i,j], vmin=0, vmax=25, cmap='jet')
        fig.colorbar(im, ax=ax[1,1], location="bottom")

        ax[0,2].set_title('CBV (ml/100g)')
        ax[0,2].imshow(cbv[i,j], vmin=0, vmax=6, cmap='jet')
        im = ax[1,2].imshow(gt_cbv[i,j], vmin=0, vmax=6, cmap='jet')
        fig.colorbar(im, ax=ax[1,2], location="bottom")

        ax[0,3].set_title('Delay (s)')
        im = ax[0,3].imshow(delay[i,j], vmin=0, vmax=3.5, cmap='jet')
        im = ax[1,3].imshow(gt_delay[i,j], vmin=0, vmax=3.5, cmap='jet')

        fig.colorbar(im, ax=ax[1,3], location="bottom")

        for x in ax.flatten():
            x.set_axis_off()
        fig.suptitle('Parameter estimation epoch: {}'.format(epoch))
        plt.tight_layout()
        plt.show()
def get_bolus_arrival_time(curve):

    curve_dt = torch.gradient(curve)[0]
    curve_dtdt = torch.gradient(curve_dt)[0]
    top_2_maximums = torch.topk(curve_dtdt, k=2)
    index_of_bolus_arrival = torch.min(top_2_maximums.indices).item()

    # plt.plot(curve.numpy(), label='curve')
    # plt.plot(10 * curve_dt.numpy(), label='first derivative')
    # plt.plot(10 * curve_dtdt.numpy(), label='second derivative')
    plt.scatter(index_of_bolus_arrival, curve.numpy()[index_of_bolus_arrival], label='Bolus arrival', c='k')
    # plt.legend()
    # plt.show()