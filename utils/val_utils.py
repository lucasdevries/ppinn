import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from einops.einops import repeat
import pandas as pd
def load_nlr_results(cbv_ml=5, sd=2):
    result = sitk.ReadImage(rf'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\nlr_results\nlr_sd_{sd}.nii')
    result = sitk.GetArrayFromImage(result)
    result = result[:,cbv_ml-1,:,:]
    density = 1.05
    constant = (100 / density) * 0.55 / 0.75
    cbv = result[0]
    cbv = cbv * constant
    mtt = result[1]
    cbf = cbv / (mtt / 60)
    delay = result[2]
    tmax = delay + 0.5 * mtt

    return {'cbf': cbf,
            'mtt': mtt,
            'cbv': cbv,
            'delay': delay,
            'tmax': tmax}

def read_dcm(folder):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def load_sygnovia_results(cbv_ml=5, sd=2):
    simulated_data_size = 32 * 7
    scan_center = 512 // 2
    simulated_data_start = scan_center - simulated_data_size // 2
    simulated_data_end = scan_center + simulated_data_size // 2
    data = {}
    data['cbf'] = read_dcm(rf'data/phantom_sygnovia_sd{sd}/CBF')
    data['cbv'] = read_dcm(rf'data/phantom_sygnovia_sd{sd}/CBV')
    data['mtt'] = read_dcm(rf'data/phantom_sygnovia_sd{sd}/MTT')
    data['tmax'] = read_dcm(rf'data/phantom_sygnovia_sd{sd}/Tmax')
    data['ttd'] = read_dcm(rf'data/phantom_sygnovia_sd{sd}/TTD')
    data['delay'] = data['tmax'] - 0.5 * data['mtt']

    for key, val in data.items():
        array = sitk.GetArrayFromImage(val)[11:]
        array = array[cbv_ml-1]
        data[key] = array[simulated_data_start:simulated_data_end, simulated_data_start:simulated_data_end]
    return data

def load_phantom_gt(cbv_ml=5, simulation_method=2):
    perfusion_values = np.empty([5, 7, 7, 4])
    cbv = [1, 2, 3, 4, 5]  # in ml / 100g
    mtt_s = [24.0, 12.0, 8.0, 6.0, 4.8, 4.0, 3.42857143]  # in seconds
    mtt_m = [t / 60 for t in mtt_s]  # in minutes
    delay = [0., 0.5, 1., 1.5, 2., 2.5, 3.]  # in seconds

    for ix, i in enumerate(cbv):
        for jx, j in enumerate(delay):
            for kx, k in enumerate(mtt_m):
                # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
                values = np.array([i, j, k, i / k])
                perfusion_values[ix, jx, kx] = values
    perfusion_values = repeat(perfusion_values, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    perfusion_values_dict = {'cbf': perfusion_values[simulation_method, cbv_ml-1, ..., 3],
                             'delay': perfusion_values[simulation_method, cbv_ml-1, ..., 1],
                             'cbv': perfusion_values[simulation_method, cbv_ml-1, ..., 0],
                             'mtt': perfusion_values[simulation_method, cbv_ml-1, ..., 2] * 60}
    perfusion_values_dict['tmax'] = perfusion_values_dict['delay'] + 0.5 * perfusion_values_dict['mtt']
    return perfusion_values_dict
def visualize(slice, case, perfusion_values, result_dict):
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    cbf_results = result_dict['cbf'].cpu().detach().numpy()
    cbv_results = result_dict['cbv'].cpu().detach().numpy()
    mtt_results = result_dict['mtt'].cpu().detach().numpy()
    delay_results = result_dict['delay'].cpu().detach().numpy()
    tmax_results = result_dict['tmax'].cpu().detach().numpy()

    isles_cbf = perfusion_values[..., 0]
    isles_cbv = perfusion_values[..., 1]
    isles_mtt = perfusion_values[..., 2]
    isles_tmax = perfusion_values[..., 3]
    isles_gt_core = perfusion_values[..., 4]
    isles_delay = perfusion_values[..., 5]

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150
    fig, ax = plt.subplots(2, 6, figsize=(12, 6))

    ax[0, 0].set_title('CBF', fontdict=font)

    im = ax[0, 0].imshow(cbf_results[0], vmin=np.percentile(cbf_results[0][cbf_results[0]>0],10), vmax=np.percentile(cbf_results[0][cbf_results[0]>0],90), cmap='jet')
    cax = ax[0, 0].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[1, 0].imshow(isles_cbf, vmin=np.percentile(isles_cbf[isles_cbf>0], 10), vmax=np.percentile(isles_cbf[isles_cbf>0],90), cmap='jet')
    cax = ax[1, 0].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 1].set_title('MTT', fontdict=font)

    im = ax[0, 1].imshow(mtt_results[0], vmin=np.percentile(mtt_results[0][mtt_results[0]>0],10), vmax=np.percentile(mtt_results[0][mtt_results[0]>0],90), cmap='jet')
    cax = ax[0, 1].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[1, 1].imshow(isles_mtt, vmin=np.percentile(isles_mtt[isles_mtt>0], 10), vmax=np.percentile(isles_mtt[isles_mtt>0],90), cmap='jet')
    cax = ax[1, 1].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 2].set_title('CBV', fontdict=font)

    im = ax[0, 2].imshow(cbv_results[0], vmin=np.percentile(cbv_results[0][cbv_results[0]>0],10), vmax=np.percentile(cbv_results[0][cbv_results[0]>0],90), cmap='jet')
    cax = ax[0, 2].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[1, 2].imshow(isles_cbv, vmin=np.percentile(isles_cbv[isles_cbv>0], 10), vmax=np.percentile(isles_cbv[isles_cbv>0],90), cmap='jet')
    cax = ax[1, 2].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 3].set_title('Tmax', fontdict=font)
    im = ax[0, 3].imshow(tmax_results[0], vmin=np.percentile(tmax_results[0][tmax_results[0]>0],10), vmax=np.percentile(tmax_results[0][tmax_results[0]>0],90), cmap='jet')
    cax = ax[0, 3].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    im = ax[1, 3].imshow(isles_tmax, vmin=np.percentile(isles_tmax[isles_tmax>0], 10), vmax=np.percentile(isles_tmax[isles_tmax>0],90), cmap='jet')
    cax = ax[1, 3].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 4].set_title('Delay', fontdict=font)
    im = ax[0, 4].imshow(delay_results[0], vmin=np.percentile(delay_results[0][delay_results[0]>0],10), vmax=np.percentile(delay_results[0][delay_results[0]>0],90), cmap='jet')
    cax = ax[0, 4].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    im = ax[1, 4].imshow(isles_delay, vmin=np.percentile(isles_delay[isles_delay>0], 10), vmax=np.percentile(isles_delay[isles_delay>0],90), cmap='jet')
    cax = ax[1, 4].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 5].set_title('Core', fontdict=font)
    ax[1, 5].set_title('Core', fontdict=font)

    im = ax[0, 5].imshow(isles_gt_core, vmin=0.01, vmax=2, cmap='jet')
    cax = ax[0, 5].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    im = ax[1, 5].imshow(isles_gt_core, vmin=0.01, vmax=2, cmap='jet')
    cax = ax[1, 5].inset_axes([0, -0.2, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    # for i in range(5):
    #     ax[1, i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    ax[0, 0].set_ylabel('PPINN', fontdict=font)
    ax[1, 0].set_ylabel('ISLES/Rapid', fontdict=font)
    # plt.tight_layout()
    wandb.log({"results_{}".format(case): plt})
    # plt.show()
    plt.close()


def plot_software_results(results_dict, phantom_dict, name='Sygno.via'):

    cbf = results_dict['cbf']
    cbv = results_dict['cbv']
    mtt = results_dict['mtt']
    tmax = results_dict['tmax']
    delay = results_dict['delay']

    gt_cbf = phantom_dict['cbf']
    gt_cbv = phantom_dict['cbv']
    gt_mtt = phantom_dict['mtt']
    gt_delay = phantom_dict['delay']
    gt_tmax = phantom_dict['tmax']


    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150


    fig, ax = plt.subplots(3, 5, figsize=(10,6))
    ax[0, 0].set_title('CBF', fontdict=font)
    ax[0, 0].imshow(cbf, vmin=0, vmax=100, cmap='jet')
    im = ax[1, 0].imshow(gt_cbf, vmin=0, vmax=100, cmap='jet')
    cax = ax[2, 0].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    ax[0, 0].set_ylabel(f'{name}', fontdict=font)
    ax[1, 0].set_ylabel('GT', fontdict=font)

    ax[0, 1].set_title('MTT', fontdict=font)
    ax[0, 1].imshow(mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    im = ax[1, 1].imshow(gt_mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    cax = ax[2, 1].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 2].set_title('CBV', fontdict=font)
    ax[0, 2].imshow(cbv, vmin=0.01, vmax=7, cmap='jet')
    im = ax[1, 2].imshow(gt_cbv, vmin=0.01, vmax=7, cmap='jet')
    cax = ax[2, 2].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 3].set_title('Delay', fontdict=font)
    ax[0, 3].imshow(delay, vmin=0.01, vmax=3.5, cmap='jet')
    im = ax[1, 3].imshow(gt_delay, vmin=0.01, vmax=3.5, cmap='jet')
    cax = ax[2, 3].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    ax[0, 4].set_title('Tmax', fontdict=font)
    ax[0, 4].imshow(tmax, vmin=0.01, vmax=15, cmap='jet')
    im = ax[1, 4].imshow(gt_tmax, vmin=0.01, vmax=15, cmap='jet')
    cax = ax[2, 4].inset_axes([0, 0.82, 1, 0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    for i in range(5):
        ax[2, i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    fig.suptitle(f'{name} vs. Phantom ground truth', fontdict=font)
    plt.tight_layout()
    os.makedirs(os.path.join(wandb.run.dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(wandb.run.dir, 'plots', f'{name}_vs_gt.png'), dpi=150)
    # plt.show()

def plot_software_results_on_axis(ax, results_dict, name='Sygno.via', title=False):

    cbf = results_dict['cbf']
    cbv = results_dict['cbv']
    mtt = results_dict['mtt']
    tmax = results_dict['tmax']
    delay = results_dict['delay']

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    # fig, ax = plt.subplots(3, 5, figsize=(10,6))
    if title:
        ax[0].set_title('CBF', fontdict=font)
        ax[1].set_title('MTT', fontdict=font)
        ax[2].set_title('CBV', fontdict=font)
        ax[3].set_title('Delay', fontdict=font)
        ax[4].set_title('Tmax', fontdict=font)

    ax[0].imshow(cbf, vmin=0, vmax=100, cmap='jet')
    ax[0].set_ylabel(f'{name}', fontdict=font)
    ax[1].imshow(mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    ax[2].imshow(cbv, vmin=0.01, vmax=7, cmap='jet')
    ax[3].imshow(delay, vmin=0.01, vmax=3.5, cmap='jet')
    ax[4].imshow(tmax, vmin=0.01, vmax=15, cmap='jet')

    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
def plot_software_difference_on_axis(ax, results_dict, gt_dict, name='Sygno.via', title=False):

    cbf = results_dict['cbf'] - gt_dict['cbf']
    cbv = results_dict['cbv'] - gt_dict['cbv']
    mtt = results_dict['mtt'] - gt_dict['mtt']
    tmax = results_dict['tmax'] - gt_dict['tmax']
    delay = results_dict['delay'] - gt_dict['delay']

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150

    # fig, ax = plt.subplots(3, 5, figsize=(10,6))
    if title:
        ax[0].set_title('CBF', fontdict=font)
        ax[1].set_title('MTT', fontdict=font)
        ax[2].set_title('CBV', fontdict=font)
        ax[3].set_title('Delay', fontdict=font)
        ax[4].set_title('Tmax', fontdict=font)

    ax[0].set_ylabel(f'{name}', fontdict=font)
    ax[0].imshow(cbf, vmin=-10, vmax=10, cmap='jet')
    ax[1].imshow(mtt, vmin=-2, vmax=2, cmap='jet')
    ax[2].imshow(cbv, vmin=-1, vmax=1, cmap='jet')
    ax[3].imshow(delay, vmin=-1, vmax=1, cmap='jet')
    ax[4].imshow(tmax, vmin=-1, vmax=1, cmap='jet')

    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
def add_colorbars_to_fig(phantom_dict, fig, ax, gt_axis):
    cbar_axis = gt_axis + 1
    gt_cbf = phantom_dict['cbf']
    gt_cbv = phantom_dict['cbv']
    gt_mtt = phantom_dict['mtt']
    gt_delay = phantom_dict['delay']
    gt_tmax = phantom_dict['tmax']

    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150
    # [x0, y0, width, height]
    im = ax[gt_axis, 0].imshow(gt_cbf, vmin=0, vmax=100, cmap='jet')
    cax = ax[cbar_axis, 0].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    ax[gt_axis, 0].set_ylabel('GT', fontdict=font)
    ax[gt_axis, 1].set_ylabel(' ', fontdict=font)

    im = ax[gt_axis, 1].imshow(gt_mtt, vmin=0.01, vmax=1.1*24, cmap='jet')
    cax = ax[cbar_axis, 1].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 2].imshow(gt_cbv, vmin=0.01, vmax=7, cmap='jet')
    cax = ax[cbar_axis, 2].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 3].imshow(gt_delay, vmin=0.01, vmax=3.5, cmap='jet')
    cax = ax[cbar_axis, 3].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 4].imshow(gt_tmax, vmin=0.01, vmax=15, cmap='jet')
    cax = ax[cbar_axis, 4].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    for i in range(5):
        ax[cbar_axis, i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    # plt.tight_layout()
def add_colorbars_to_fig_difference(fig, ax, gt_axis):
    # cbf = resdict['cbf']
    # cbv = resdict['cbv']
    # mtt = resdict['mtt']
    # delay = resdict['delay']
    # tmax = resdict['tmax']
    cbar_axis = gt_axis + 1
    font = {'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 15,
            }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["figure.dpi"] = 150
    # [x0, y0, width, height]
    im = ax[gt_axis, 0].imshow(np.zeros((224,224)), vmin=-10, vmax=10, cmap='jet')
    cax = ax[cbar_axis, 0].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g/min', fontdict=font)
    bar.ax.tick_params(labelsize=14)
    ax[gt_axis, 0].set_ylabel('GT', fontdict=font)
    ax[gt_axis, 1].set_ylabel(' ', fontdict=font)
    im = ax[gt_axis, 1].imshow(np.zeros((224,224)), vmin=-2, vmax=2, cmap='jet')
    cax = ax[cbar_axis, 1].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 2].imshow(np.zeros((224,224)), vmin=-1, vmax=1, cmap='jet')
    cax = ax[cbar_axis, 2].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('ml/100g', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 3].imshow(np.zeros((224,224)), vmin=-1, vmax=1, cmap='jet')
    cax = ax[cbar_axis, 3].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    im = ax[gt_axis, 4].imshow(np.zeros((224,224)), cmap='jet')
    cax = ax[cbar_axis, 4].inset_axes([0,
                                       ax[gt_axis,0].get_position().y0+0.7,
                                       1,
                                       0.1])
    bar = fig.colorbar(im, cax=cax, orientation="horizontal")
    bar.outline.set_color('black')
    bar.set_label('seconds', fontdict=font)
    bar.ax.tick_params(labelsize=14)

    for i in range(5):
        ax[cbar_axis, i].set_axis_off()
    for x in ax.flatten():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    # plt.tight_layout()

# def log_software_results(results_dict, gt_dict, name='Sygno.via'):
#
#     min_cbf, max_cbf = 0, 150
#     wandb.log({f"cbf_{name}_mse": np.mean((results_dict['cbf'] - gt_dict['cbf']) ** 2),
#                f"cbv_{name}_mse": np.mean((results_dict['cbv'] - gt_dict['cbv']) ** 2),
#                f"mtt_{name}_mse": np.mean((results_dict['mtt'] - gt_dict['mtt']) ** 2),
#                f"delay_{name}_mse": np.mean((results_dict['delay'] - gt_dict['delay']) ** 2),
#                f"tmax_{name}_mse": np.mean((results_dict['tmax'] - gt_dict['tmax']) ** 2),
#                f"cbf_{name}_mae": np.abs((results_dict['cbf'] - gt_dict['cbf'])),
#                f"cbv_{name}_mae": np.abs((results_dict['cbv'] - gt_dict['cbv'])),
#                f"mtt_{name}_mae": np.abs((results_dict['mtt'] - gt_dict['mtt'])),
#                f"delay_{name}_mae": np.abs((results_dict['delay'] - gt_dict['delay'])),
#                f"tmax_{name}_mae": np.abs((results_dict['tmax'] - gt_dict['tmax'])),
#                f"cbf_{name}_mean": np.mean((results_dict['cbf'] - gt_dict['cbf'])),
#                f"cbv_{name}_mean": np.mean((results_dict['cbv'] - gt_dict['cbv'])),
#                f"mtt_{name}_mean": np.mean((results_dict['mtt'] - gt_dict['mtt'])),
#                f"delay_{name}_mean": np.mean((results_dict['delay'] - gt_dict['delay'])),
#                f"tmax_{name}_mean": np.mean((results_dict['tmax'] - gt_dict['tmax']))
#                })

def log_software_results(results, cbv_ml):
    for metric in ['mse', 'mae', 'me']:
        table = []
        for key in ['cbf', 'cbv', 'mtt', 'delay', 'tmax']:
            if metric == 'mse':
                table.append([cbv_ml, f"{key}",
                              np.mean((results['nlr'][key] - results['gt'][key]) ** 2),
                              np.mean((results['sygnovia'][key] - results['gt'][key]) ** 2),
                              np.mean((results['ppinn'][key] - results['gt'][key]) ** 2)])
            elif metric == 'mae':
                table.append([cbv_ml, f"{key}",
                              np.mean(np.abs(results['nlr'][key] - results['gt'][key])),
                              np.mean(np.abs(results['sygnovia'][key] - results['gt'][key])),
                              np.mean(np.abs(results['ppinn'][key] - results['gt'][key]))])
            elif metric == 'me':
                table.append([cbv_ml, f"{key}",
                              np.mean(results['nlr'][key] - results['gt'][key]),
                              np.mean(results['sygnovia'][key] - results['gt'][key]),
                              np.mean(results['ppinn'][key] - results['gt'][key])])
            else:
                raise NotImplementedError('Not implemented')
        columns = ['cbv_ml', 'parameter', 'NLR', 'Sygnovia', 'PPINN']
        df = pd.DataFrame(columns=['cbv_ml', 'parameter', 'NLR', 'Sygnovia', 'PPINN'], data=table)
        wandb_table = wandb.Table(data=df)
        # wandb_table = wandb.Table(data=table, columns=columns)
        wandb.log({f'table_{metric}': wandb_table})
def drop_edges(results):
    skip_rows = sorted(list(range(-2, 226, 32)) + list(range(-1, 226, 32)) + list(range(0, 226, 32)) + list(range(1, 226, 32)))
    skip_rows = skip_rows[2:-2]
    for k1 in results.keys():
        for k2 in ['cbf', 'cbv', 'mtt', 'delay', 'tmax']:
            results[k1][k2] = np.delete(results[k1][k2], skip_rows, axis=0)
            results[k1][k2] = np.delete(results[k1][k2], skip_rows, axis=1)
    return results

def drop_unphysical(results):
    for k1 in results.keys():
        results[k1]['cbf'] = np.clip(results[k1]['cbf'], a_min=0 , a_max=150)
        results[k1]['cbv'] = np.clip(results[k1]['cbv'], a_min=0, a_max=10)
        results[k1]['mtt'] = np.clip(results[k1]['mtt'], a_min=0 , a_max=30)
        results[k1]['delay'] = np.clip(results[k1]['delay'], a_min=0 , a_max=10)
        results[k1]['tmax'] = np.clip(results[k1]['tmax'], a_min=0 , a_max=15)
    return results

def plot_results(results):
    fig, ax = plt.subplots(5, 5, figsize=(14, 14))
    plot_software_results_on_axis(ax[0], results['sygnovia'], name='Sygno.via', title=True)
    plot_software_results_on_axis(ax[1], results['nlr'], name='NLR')
    plot_software_results_on_axis(ax[2], results['ppinn'], name='PPINN')
    plot_software_results_on_axis(ax[3], results['gt'], name='GT')
    add_colorbars_to_fig(results['gt'], fig, ax, 3)
    # plt.tight_layout()
    os.makedirs(os.path.join(wandb.run.dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(wandb.run.dir, 'plots', f'software_vs_gt.png'), dpi=150)
    wandb.log({"results_compare": plt})

    fig, ax = plt.subplots(4, 5, figsize=(14, 14))
    add_colorbars_to_fig_difference(fig, ax, 2)
    plot_software_difference_on_axis(ax[0], results['sygnovia'], results['gt'], name='Sygno.via', title=True)
    plot_software_difference_on_axis(ax[1], results['nlr'], results['gt'], name='NLR')
    plot_software_difference_on_axis(ax[2], results['ppinn'], results['gt'], name='PPINN')
    # plt.tight_layout()
    os.makedirs(os.path.join(wandb.run.dir, 'plots'), exist_ok=True)
    plt.savefig(os.path.join(wandb.run.dir, 'plots', f'software_vs_gt_difference.png'), dpi=150)
    wandb.log({"results_compare_difference": plt})
    # plt.show()
