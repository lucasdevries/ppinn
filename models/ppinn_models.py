import torch
import torch.nn as nn
import numpy as np
from models.MLP import MLP
from utils.train_utils import AverageMeter
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import wandb
from torch.optim.swa_utils import AveragedModel, SWALR
from torchcontrib.optim import SWA
class PPINN(nn.Module):
    def __init__(self,
                 config,
                 shape_in,
                 perfusion_values,
                 n_inputs=1,
                 std_t=1):
        super(PPINN, self).__init__()
        self.config = config
        self.PID = os.getpid()
        self.logger = logging.getLogger(str(self.PID))
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda:{}".format(self.config.gpu_device))
            torch.cuda.set_device("cuda:{}".format(self.config.gpu_device))
            self.logger.info("Operation will be on *****GPU-CUDA{}***** ".format(self.config.gpu_device))
        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")

        self.lw_data, self.lw_res, self.lw_bc = (0, 0, 0)
        self.optimizer = None
        self.scheduler = None
        self.milestones = [int(0.5*config.epochs), int(0.8 * config.epochs)]
        self.interpolator = None
        self.var_list = None
        self.shape_in = shape_in
        self.std_t = std_t
        self.neurons_out = 1
        self.perfusion_values = perfusion_values
        # initialize flow parameters
        self.log_domain = config.log_domain
        self.delay_type = self.config.delay_type
        self.cbf_type = self.config.cbf_type
        self.mtt_type = self.config.mtt_type
        self.set_delay_parameter()
        self.set_cbf_parameter()
        self.set_mtt_parameter()

        n_layers = config.n_layers
        n_units = config.n_units
        lr = config.lr
        loss_weights = (config.lw_data, config.lw_res, 0)
        bn = config.bn

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
        self.set_params_to_domain()
        self.set_device(self.device)
        self.float()


    def forward_NNs(self, t):
        t = t.unsqueeze(-1)
        c_tissue = self.NN_tissue(t)
        c_aif = self.NN_aif(t)
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
        if self.delay_type == 'learned':
            cbf, mtt, delay = self.get_ode_params()
        elif self.delay_type == 'fixed':
            cbf, mtt, _ = self.get_ode_params()
            delay = self.get_delay()
            delay.requires_grad = False
        else:
            raise NotImplementedError('Delay type not implemented...')

        # Get AIF NN output:
        t = t.view(1, 1, 1, 1, steps, 1)
        t = t.expand(*self.shape_in, steps, 1)
        t = t.detach()

        t.requires_grad = False
        delay = delay.expand(*self.shape_in, steps, 1)
        c_aif_a = self.NN_aif(t - delay / self.std_t)
        c_aif_b = self.NN_aif(t - delay / self.std_t - mtt / self.std_t)

        residual = c_tissue_dt - cbf * (c_aif_a - c_aif_b)

        return c_aif, c_tissue, residual

    def set_loss_weights(self, loss_weights):
        loss_weights = torch.tensor(loss_weights)
        self.lw_data, self.lw_res, self.lw_bc = loss_weights
        self.lw_data.to(self.device)
        self.lw_res.to(self.device)
        self.lw_bc.to(self.device)

    def set_lr(self, lr):
        # base_opt = torch.optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = SWA(base_opt, swa_start=2000, swa_freq=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=self.milestones,
                                                              gamma=0.5)

    def set_device(self, device):
        self.to(device)

    def set_params_to_domain(self):
        for name, param in self.named_parameters():
            if 'flow' in name:
                param.data = torch.log(param.data) if self.log_domain else param.data

    def set_delay_parameter(self):
        if self.delay_type == 'learned':
            self.flow_t_delay = torch.nn.Parameter(torch.rand(*self.shape_in, 1, 1))
        elif self.delay_type == 'fixed':
            if self.log_domain:
                self.flow_t_delay = torch.log(self.perfusion_values[..., 1] / 3)
            else:
                self.flow_t_delay = self.perfusion_values[..., 1] / 3
            self.flow_t_delay = self.flow_t_delay.view(*self.flow_t_delay.shape, 1, 1)
            self.flow_t_delay = self.flow_t_delay.to(self.device)
        else:
            raise NotImplementedError('Delay type not implemented...')
    def set_cbf_parameter(self):
        low = 0
        high = 100 / (69.84 * 60)
        density = 1.05
        constant = (100 / density) * 0.55 / 0.75
        if self.cbf_type == 'learned':
            self.flow_cbf = torch.nn.Parameter(torch.rand(*self.shape_in, 1) * high)
            torch.nn.init.uniform_(self.flow_cbf, 0, high)
        elif self.cbf_type == 'fixed':
            if self.log_domain:
                self.flow_cbf = torch.log(self.perfusion_values[..., 3] / (constant * 60))
            else:
                self.flow_cbf = self.perfusion_values[..., 3] / (constant * 60)
            self.flow_cbf = self.flow_cbf.view(*self.flow_cbf.shape, 1)
            self.flow_cbf = self.flow_cbf.to(self.device)
        else:
            raise NotImplementedError('Delay type not implemented...')

    def set_mtt_parameter(self):
        if self.mtt_type == 'learned':
            self.flow_mtt = torch.nn.Parameter(torch.rand(*self.shape_in, 1, 1))
            torch.nn.init.uniform_(self.flow_mtt, 0.75, 1)
        elif self.mtt_type == 'fixed':
            if self.log_domain:
                self.flow_mtt = torch.log(self.perfusion_values[..., 2]*60 / 24)
            else:
                self.flow_mtt = self.perfusion_values[..., 2]*60 / 24
            self.flow_mtt = self.flow_mtt.view(*self.flow_cbf.shape, 1)
            self.flow_mtt = self.flow_mtt.to(self.device)
        else:
            raise NotImplementedError('Delay type not implemented...')
    def get_ode_params(self):
        if self.log_domain:
            return [torch.exp(self.flow_cbf), 24 * torch.exp(self.flow_mtt), 3 * torch.exp(self.flow_t_delay)]

        else:
            return [self.flow_cbf, 24 * self.flow_mtt, 3 * self.flow_t_delay]

    def get_delay(self, seconds=True):
        _, _, delay = self.get_ode_params()
        if seconds:
            return delay
        else:
            return delay / 60

    def get_mtt(self, seconds=True):
        _, mtt, _ = self.get_ode_params()
        if seconds:
            return mtt.squeeze(-1)
        else:
            return mtt.squeeze(-1) / 60

    def get_cbf(self, seconds=True):
        density = 1.05
        constant = (100 / density) * 0.55 / 0.75
        constant = torch.as_tensor(constant).to(self.device)
        flow, _, _ = self.get_ode_params()
        if seconds:
            return constant * flow
        else:
            return constant * flow * 60

    def fit(self,
            data_time,
            data_aif,
            data_curves,
            data_collopoints,
            data_boundary,
            gt,
            batch_size,
            epochs):

        collopoints_dataloader = DataLoader(data_collopoints, batch_size=batch_size, shuffle=True, drop_last=True)

        for ep in tqdm(range(self.current_iteration + 1, self.current_iteration + epochs + 1)):
            epoch_aif_loss = AverageMeter()
            epoch_tissue_loss = AverageMeter()
            epoch_residual_loss = AverageMeter()

            for batch_collopoints in collopoints_dataloader:
                batch_time = data_time
                batch_aif = data_aif
                batch_curves = data_curves
                batch_boundary = data_boundary
                loss_aif, loss_tissue, loss_residual = self.optimize(batch_time,
                                                         batch_aif,
                                                         batch_curves,
                                                         batch_boundary,
                                                         batch_collopoints)

                epoch_aif_loss.update(loss_aif.item(), len(batch_time))
                epoch_tissue_loss.update(loss_tissue.item(), len(batch_time))
                epoch_residual_loss.update(loss_residual.item(), len(batch_collopoints))

            self.scheduler.step()

            metrics = {"aif_loss": epoch_aif_loss.avg,
                       "tissue_loss": epoch_tissue_loss.avg,
                       "residual_loss": epoch_residual_loss.avg,
                       "lr": self.optimizer.param_groups[0]['lr'],
                       }
            validation_metrics = self.validate()
            metrics.update(validation_metrics)

            wandb.log(metrics, step=self.current_iteration)

            if ep % self.config.plot_params_every == 0:
                self.plot_params(0, 0, gt, ep)
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
        loss_aif, loss_tissue, loss_residual = 999, 999, 999

        if self.current_iteration < 0:
            if self.lw_res:
                # compute residual loss
                c_aif, c_tissue, residual = self.forward_complete(batch_collopoints)
                loss_residual = self.__loss_residual(residual)
                loss += self.lw_res * loss_residual
                # loss_aif, loss_tissue = 0, 0
        else:
            if self.lw_data:
                # compute data loss
                if self.current_iteration > 6000:
                    c_aif, c_tissue = self.forward_NNs(batch_time)
                else:
                    c_aif, c_tissue = self.forward_NNs(batch_time)
                loss_aif, loss_tissue = self.__loss_data(batch_aif, batch_curves, c_aif, c_tissue)
                loss += self.lw_data * (loss_aif + loss_tissue)
            if self.lw_res:
                # compute residual loss
                c_aif, c_tissue, residual = self.forward_complete(batch_collopoints)
                loss_residual = self.__loss_residual(residual)
                loss += self.lw_res * loss_residual
            else:
                raise NotImplementedError('No loss calculated... all weights at zero?')

        if np.isnan(float(loss.item())):
            raise ValueError('Loss is nan during training...')

        loss.backward()
        self.optimizer.step()
        return loss_aif, loss_tissue, loss_residual

    def validate(self):

        # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'

        gt_cbv = self.perfusion_values[..., 0]
        gt_delay = self.perfusion_values[..., 1]
        gt_mtt = self.perfusion_values[..., 2] * 60
        gt_cbf = self.perfusion_values[..., 3]

        [gt_cbv, gt_cbf, gt_mtt, gt_delay] = [torch.as_tensor(x).to(self.device)
                                              for x in [gt_cbv, gt_cbf, gt_mtt, gt_delay]]
        cbf = self.get_cbf(seconds=False).squeeze(-1)
        mtt = self.get_mtt(seconds=True).squeeze(-1)
        mtt_min = self.get_mtt(seconds=False).squeeze(-1)
        delay = self.get_delay(seconds=True).squeeze(-1).squeeze(-1)
        cbv = cbf * mtt_min
        cbv_mse = torch.nn.functional.mse_loss(cbv, gt_cbv).item()
        cbf_mse = torch.nn.functional.mse_loss(cbf, gt_cbf).item()
        mtt_mse = torch.nn.functional.mse_loss(mtt, gt_mtt).item()
        delay_mse = torch.nn.functional.mse_loss(delay, gt_delay).item()
        return {'cbv_mse': cbv_mse,
                'cbf_mse': cbf_mse,
                'mtt_mse': mtt_mse,
                'delay_mse': delay_mse}

    def save_parameters(self):
        # Save NNs
        torch.save(self.state_dict(), os.path.join(wandb.run.dir, 'model.pth.tar'))
        torch.save(self.NN_tissue.state_dict(), os.path.join(wandb.run.dir, 'NN_tissue.pth.tar'))
        torch.save(self.NN_aif.state_dict(), os.path.join(wandb.run.dir, 'NN_aif.pth.tar'))
        # Save parameters
        torch.save(self.flow_mtt, os.path.join(wandb.run.dir, 'flow_mtt.pth.tar'))
        torch.save(self.flow_cbf, os.path.join(wandb.run.dir, 'flow_cbf.pth.tar'))
        torch.save(self.flow_t_delay, os.path.join(wandb.run.dir, 'flow_t_delay.pth.tar'))
        # Save parameter data
        for name, param in self.named_parameters():
            if 'flow_' in name:
                parameter_data = param.data.cpu().numpy()
                with open(os.path.join(wandb.run.dir, f'{name}.npy'), 'wb') as f:
                    np.save(f, parameter_data)

    def __loss_data(self, aif, curves, c_aif, c_tissue):
        # reshape the ground truth
        aif = aif.expand(*c_aif.shape)
        # solution loss
        loss_aif = F.mse_loss(aif, c_aif)
        loss_tissue = F.mse_loss(curves, c_tissue)
        # TODO maybe implement derivative loss here
        return loss_aif, loss_tissue

    def __loss_interpolation(self, aif, curves, output):
        # TODO implement loss that uses C_aif(t-MTT) estimation and compares to interpolated version of AIF
        pass

    def __loss_residual(self, residual):
        loss_r = torch.mean(torch.square(residual))
        return loss_r

    def __loss_bc(self, output):
        _, _, _, _, _ = output
        # TODO implement
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
        # cbf = torch.clip(cbf, min=0, max=125)
        cbv = cbf * mtt_min
        # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
        gt_cbv = perfusion_values[..., 0]
        gt_delay = perfusion_values[..., 1]
        gt_mtt = perfusion_values[..., 2] * 60
        gt_cbf = perfusion_values[..., 3]

        cbf_min, cbf_max = 0.9*torch.min(gt_cbf).item(), 1.1*torch.max(gt_cbf).item()

        [cbf, mtt, cbv, gt_cbf, gt_mtt, gt_cbv, delay] = [x.detach().cpu().numpy() for x in
                                                          [cbf, mtt, cbv, gt_cbf, gt_mtt, gt_cbv, delay]]
        i, j = 0, 0

        fig, ax = plt.subplots(2, 4)

        ax[0, 0].set_title('CBF (ml/100g/min)')
        ax[0, 0].imshow(cbf[i, j], vmin=cbf_min, vmax=cbf_max, cmap='jet')
        im = ax[1, 0].imshow(gt_cbf[i, j], vmin=cbf_min, vmax=cbf_max, cmap='jet')
        fig.colorbar(im, ax=ax[1, 0], location="bottom")

        ax[0, 1].set_title('MTT (s)')
        ax[0, 1].imshow(mtt[i, j], vmin=0.01, vmax=24, cmap='jet')
        im = ax[1, 1].imshow(gt_mtt[i, j], vmin=0.01, vmax=24, cmap='jet')
        fig.colorbar(im, ax=ax[1, 1], location="bottom")

        ax[0, 2].set_title('CBV (ml/100g)')
        ax[0, 2].imshow(cbv[i, j], vmin=0.01, vmax=7, cmap='jet')
        im = ax[1, 2].imshow(gt_cbv[i, j], vmin=0.01, vmax=7, cmap='jet')
        fig.colorbar(im, ax=ax[1, 2], location="bottom")

        ax[0, 3].set_title('Delay (s)')
        im = ax[0, 3].imshow(delay[i, j], vmin=0.01, vmax=3.5, cmap='jet')
        im = ax[1, 3].imshow(gt_delay[i, j], vmin=0.01, vmax=3.5, cmap='jet')

        fig.colorbar(im, ax=ax[1, 3], location="bottom")

        for x in ax.flatten():
            x.set_axis_off()
        fig.suptitle('Parameter estimation epoch: {}'.format(epoch))
        plt.tight_layout()
        wandb.log({"parameters": plt}, step=epoch)
        # plt.show()
        plt.close()