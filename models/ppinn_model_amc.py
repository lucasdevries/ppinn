import torch
import torch.nn as nn
import numpy as np
from models.MLP_amc import MLP
from utils.train_utils import AverageMeter, weightConstraint
from utils.val_utils import visualize_amc
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import wandb
import SimpleITK as sitk
from utils.val_utils import load_nlr_results, plot_curves_at_epoch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["mathtext.fontset"] = 'cm'

class PPINN_amc(nn.Module):
    def __init__(self,
                 config,
                 shape_in,
                 # perfusion_values,
                 n_inputs=1,
                 std_t=1,
                 original_data_shape = None,
                 original_indices = None
                 ):
        super(PPINN_amc, self).__init__()
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
        self.milestones = [500]
        self.interpolator = None
        self.var_list = None
        self.shape_in = shape_in
        self.original_data_shape = original_data_shape
        self.original_data_indices = original_indices
        # self.perfusion_values = perfusion_values
        self.std_t = std_t
        self.neurons_out = 1
        # initialize flow parameters
        self.log_domain = config.log_domain
        self.delay_type = self.config.delay_type
        self.cbf_type = self.config.cbf_type
        self.mtt_type = self.config.mtt_type
        self.set_delay_parameter()
        self.set_cbf_parameter()
        self.set_mtt_parameter()
        self.constraint = weightConstraint()
        self.min_aif_loss = 100
        n_layers = config.n_layers
        n_units = config.n_units
        lr = config.lr
        loss_weights = (config.lw_data, config.lw_res, config.lw_curves)
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


    def forward_NNs(self, t_tissue, t_aif):
        # Used for the data loss only
        t_tissue = t_tissue.unsqueeze(-1)
        t_aif = t_aif.unsqueeze(-1)

        c_tissue = self.NN_tissue(t_tissue)
        c_aif = self.NN_aif(t_aif)
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
        t = t.view(1, 1, 1, steps, 1)
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
        self.lw_data, self.lw_res, self.lw_curves = loss_weights
        self.lw_data.to(self.device)
        self.lw_res.to(self.device)
        self.lw_curves.to(self.device)

    def set_lr(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        # TODO
        # print('USING ADAMW dont forget')

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
            torch.nn.init.uniform_(self.flow_t_delay, 0.5, 1)
        else:
            raise NotImplementedError('Delay type not implemented...')
    def set_cbf_parameter(self):
        low = 0
        high = 100 / (69.84 * 60)
        density = 1.05
        constant = (100 / density) * 0.55 / 0.75
        if self.cbf_type == 'learned':
            self.flow_cbf = torch.nn.Parameter(torch.rand(*self.shape_in, 1) * high)
            torch.nn.init.uniform_(self.flow_cbf, 0.5*high, high)
        else:
            raise NotImplementedError('Delay type not implemented...')

    def set_mtt_parameter(self):
        if self.mtt_type == 'learned':
            self.flow_mtt = torch.nn.Parameter(torch.rand(*self.shape_in, 1, 1))
            torch.nn.init.uniform_(self.flow_mtt, 0.75, 1)
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
            slice,
            data_dict,
            batch_size,
            epochs,
            case):
        # Here we should use the corrext timing
        data_time = data_dict['time'].to(self.device)[slice]
        data_aif_time = data_dict['aif_time'].to(self.device)
        data_aif = data_dict['aif'].to(self.device)
        data_curves = data_dict['curves'][slice:slice+1]
        data_curves = data_curves[0][self.original_data_indices].unsqueeze(1).unsqueeze(0)
        data_curves = data_curves.to(self.device)

        data_collopoints = data_dict['coll_points'].to(self.device)
        data_boundary = data_dict['bound'].to(self.device)
        brainmask = data_dict['brainmask']
        collopoints_dataloader = DataLoader(data_collopoints, batch_size=batch_size, shuffle=True, drop_last=True)
        for ep in range(self.current_iteration + 1, self.current_iteration + epochs + 1):
            epoch_aif_loss = AverageMeter()
            epoch_tissue_loss = AverageMeter()
            epoch_residual_loss = AverageMeter()


            for batch_collopoints in collopoints_dataloader:
                batch_time = data_time
                batch_aif_time = data_aif_time
                batch_aif = data_aif
                batch_curves = data_curves
                batch_boundary = data_boundary

                loss_aif, loss_tissue, loss_residual = self.optimize(batch_time,
                                                         batch_aif,
                                                         batch_curves,
                                                         batch_boundary,
                                                         batch_collopoints,
                                                         batch_aif_time)

                epoch_aif_loss.update(loss_aif.item(), len(batch_time))
                epoch_tissue_loss.update(loss_tissue.item(), len(batch_time))
                epoch_residual_loss.update(loss_residual.item(), len(batch_collopoints))

                self.min_aif_loss = loss_aif.item()

            self.scheduler.step()
            metrics = {f"aif_loss_{case}_{slice}": epoch_aif_loss.avg,
                       f"tissue_loss_{case}_{slice}": epoch_tissue_loss.avg,
                       f"residual_loss_{case}_{slice}": epoch_residual_loss.avg,
                       f"lr_{case}_{slice}": self.optimizer.param_groups[0]['lr'],
                       }
            wandb.log(metrics, step=self.current_iteration)

            # if ep%5==0:
            #     plot_curves_at_epoch(data_dict, data_curves, self.device, self.forward_NNs, ep, case,slice, plot_estimates=True)
            #     plot_curves_at_epoch(data_dict, data_curves, self.device, self.forward_NNs, ep, case,slice, plot_estimates=False)

            self.current_iteration += 1

        # get results
        # self.original_data_indices = original_indices
        cbf = torch.zeros(self.original_data_shape).to(self.device)
        mtt = torch.zeros(self.original_data_shape).to(self.device)
        mtt_min = torch.zeros(self.original_data_shape).to(self.device)
        delay = torch.zeros(self.original_data_shape).to(self.device)

        cbf[self.original_data_indices] = self.get_cbf(seconds=False).squeeze()
        mtt[self.original_data_indices] = self.get_mtt(seconds=True).squeeze()
        mtt_min[self.original_data_indices] = self.get_mtt(seconds=False).squeeze()
        delay[self.original_data_indices] = self.get_delay(seconds=True).squeeze()

        cbv = cbf * mtt_min
        tmax = delay + 0.5*mtt

        return {'cbf': cbf,
                'cbv': cbv,
                'mtt': mtt,
                'delay': delay,
                'tmax': tmax}

    def optimize(self,
                 batch_time,
                 batch_aif,
                 batch_curves,
                 batch_boundary,
                 batch_collopoints,
                 batch_aif_time
                 ):
        self.train()
        self.optimizer.zero_grad()

        # batch_time = batch_time.to(self.device)
        # batch_aif = batch_aif.to(self.device)
        # batch_curves = batch_curves.to(self.device)
        # batch_boundary = batch_boundary.to(self.device)
        # batch_collopoints = batch_collopoints.to(self.device)
        batch_time.requires_grad = True
        batch_collopoints.requires_grad = True
        batch_boundary.requires_grad = True
        loss = torch.as_tensor(0.).to(self.device)
        loss_aif, loss_tissue, loss_residual = 999, 999, 999

        if self.lw_data:
            # compute data loss
            c_aif, c_tissue = self.forward_NNs(batch_time, batch_aif_time)
            loss_aif, loss_tissue = self.__loss_data(batch_aif, batch_curves, c_aif, c_tissue)
            loss += self.lw_data * (loss_aif + self.lw_curves*loss_tissue)
        # if self.lw_res:
        #     # compute residual loss
        #     c_aif, c_tissue, residual = self.forward_complete(batch_collopoints)
        #     loss_residual = self.__loss_residual(residual)
        #     loss += self.lw_res * loss_residual
        if self.lw_res:
            # compute residual loss
            c_aif, c_tissue, residual = self.forward_complete(batch_collopoints)
            loss_residual = self.__loss_residual(residual)
            loss += self.lw_res * loss_residual
        else:
            raise NotImplementedError('No loss calculated... all weights at zero?')

        if np.isnan(float(loss.item())):
            print(loss_aif, loss_tissue, loss_residual)
            raise ValueError('Loss is nan during training...')

        if loss_aif < 0.0015:
            # if self.min_aif_loss < 0.0015:
            for param in self.NN_aif.parameters():
                param.requires_grad = False

        loss.backward()
        self.optimizer.step()
        return loss_aif, loss_tissue, loss_residual


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

    # def __loss_residual(self, residual):
    #     inter = torch.mean(torch.square(residual), dim=-1)
    #     delays = self.get_delay()[..., 0, 0]
    #
    #     delayed_voxels = delays > 1.5
    #     healthy_voxels = delays <= 1.5
    #
    #     loss_delayed = torch.mean(inter[delayed_voxels])
    #     loss_healthy = torch.mean(inter[healthy_voxels])
    #
    #     if torch.isnan(loss_delayed):
    #         loss_delayed = torch.as_tensor(0.).to(self.device)
    #     if torch.isnan(loss_healthy):
    #         loss_healthy = torch.as_tensor(0.).to(self.device)
    #
    #     return 0.5*loss_delayed + 0.5*loss_healthy
    # def __loss_residual(self, residual):
    #     inter = torch.mean(torch.square(residual), dim=-1)
    #
    #     loss_high = torch.mean(inter[inter<=torch.quantile(inter, 0.33, dim=1)])
    #     loss_med = torch.mean(inter[(inter<torch.quantile(inter, 0.66, dim=1)) & (inter>torch.quantile(inter, 0.33, dim=1))])
    #     loss_low = torch.mean(inter[inter>=torch.quantile(inter, 0.66, dim=1)])
    #
    #     return loss_high + loss_med + loss_low
    # def __loss_residual(self, residual):
    #     inter = torch.mean(torch.square(residual), dim=-1)
    #     loss_1= torch.mean(inter[inter<=torch.quantile(inter, 0.1, dim=1)])
    #     loss_2 = torch.mean(inter[(inter>torch.quantile(inter, 0.1, dim=1)) & (inter<=torch.quantile(inter, 0.2, dim=1))])
    #     loss_3 = torch.mean(inter[(inter>torch.quantile(inter, 0.2, dim=1)) & (inter<=torch.quantile(inter, 0.3, dim=1))])
    #     loss_4 = torch.mean(inter[(inter>torch.quantile(inter, 0.3, dim=1)) & (inter<=torch.quantile(inter, 0.4, dim=1))])
    #     loss_5 = torch.mean(inter[(inter>torch.quantile(inter, 0.4, dim=1)) & (inter<=torch.quantile(inter, 0.5, dim=1))])
    #
    #     loss_6 = torch.mean(inter[(inter>torch.quantile(inter, 0.5, dim=1)) & (inter<=torch.quantile(inter, 0.6, dim=1))])
    #     loss_7 = torch.mean(inter[(inter>torch.quantile(inter, 0.6, dim=1)) & (inter<=torch.quantile(inter, 0.7, dim=1))])
    #     loss_8 = torch.mean(inter[(inter>torch.quantile(inter, 0.7, dim=1)) & (inter<=torch.quantile(inter, 0.8, dim=1))])
    #     loss_9 = torch.mean(inter[(inter>torch.quantile(inter, 0.8, dim=1)) & (inter<=torch.quantile(inter, 0.9, dim=1))])
    #     loss_10 = torch.mean(inter[inter>torch.quantile(inter, 0.9, dim=1)])
    #     return 0.5*(0.33*loss_8 + 0.33*loss_9 + 0.33*loss_10) + 0.5*torch.mean(torch.square(residual))
    #     return 0.1*loss_1 + 0.1*loss_2 + 0.1*loss_3 + 0.1*loss_4 + 0.1*loss_5 + 0.1*loss_6 + 0.1*loss_7 + 0.1*loss_8 + 0.1*loss_9 + 0.1*loss_9 + 0.1*loss_10
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