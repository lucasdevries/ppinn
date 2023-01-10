import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np
import os, glob
from utils.val_utils import drop_edges, drop_unphysical, drop_edges_per_method, drop_unphysical_per_method
from utils.val_utils import load_sygnovia_results, load_nlr_results, load_phantom_gt, load_sygnovia_results_amc
from utils.plot_utils import bland_altman_plot
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SimpleITK as sitk
from einops.einops import rearrange, repeat
from tqdm import tqdm
import pandas as pd
from matplotlib import cm
font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 11,
        }
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["figure.dpi"] = 150
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['legend.title_fontsize'] = 12

occlusion_side = {'C102': 'L',
                  'C103': 'L',
                  'C104': 'L',
                  'C105': 'R',
                  'C106': 'L',
                  'C107': 'R',
                  'C108': 'L',
                  'C109': 'R',
                  'C110': 'R',
                  'C111': 'R',
                  'C112': 'L',
                  'C113': 'L',
                  'C114': 'L',
                  'C115': 'R',
                  'C116': 'R'}

min_vals = {'cbf': 0.01, 'mtt': 0.01, 'cbv': 0.1, 'delay': 0.01, 'tmax': 0.01, 'dwi_seg':0.01}
max_vals = {'cbf': 100, 'mtt': 18, 'cbv': 4, 'delay': 4.5, 'tmax': 10, 'dwi_seg':1}
param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax', 'dwi_seg': 'DWI seg.'}

def plot_slice(case, method='SV'):
    if method == 'PPINN':
        base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    elif method == 'SPPINN':
        base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_sppinn_amcctp"
    elif method == 'SV':
        base = r"D:\PPINN_patient_data\sygnovia"
    else:
        raise NotImplementedError('Method not implemented.')

    if method == 'SV':
        results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii.gz")),
                   'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii.gz")),
                   'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii.gz")),
                   'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii.gz")),
                   'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii.gz")),
                   'dwi_seg': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                         "DWI_seg_registered_corrected.nii.gz")),
                   'contra': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                         "DWI_seg_registered_contralateral.nii.gz"))
                   }
    else:
        results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                   'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                   'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                   'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                   'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                   'vesselmask': sitk.ReadImage(os.path.join(base, case, "vesselmask.nii")),
                   'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                   'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz")),
                   'contra': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                         "DWI_seg_registered_contralateral.nii.gz"))
                   }

    spacing = results['dwi_seg'].GetSpacing()

    results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}
    if method == 'SV':
        empty = np.zeros_like(results['mtt'])
        empty[results['mtt'] >= 0] = 1
        results['vesselmask'] = empty

    volume = np.round(np.sum(results['dwi_seg']) * np.product(spacing) / 1000, 1)

    results['healthy'] = results['contra'] * results['vesselmask']
    results['infarcted'] = results['dwi_seg'] * results['vesselmask']
    cmap = mpl.cm.get_cmap('jet').copy()
    cmap.set_under(color='black')

    fig = plt.figure(figsize=(12, 2.5))
    outer = gridspec.GridSpec(1, 6, wspace=0.1, hspace=0.1)

    for i, key in zip(range(6), ['cbf', 'mtt', 'cbv', 'delay', 'tmax', 'dwi_seg']):
        # kwargs = {'bins': 100, 'density': True, 'histtype': 'step', 'range': (min_vals[key], max_vals[key] * 2)}
        ax = plt.Subplot(fig, outer[i])
        ax.set_title(f'{param_title[key]}', size=14, pad=5)
        if key == 'dwi_seg':
            ax.imshow(results[key][18], vmin=min_vals[key], vmax=max_vals[key], cmap='Greys_r')
        else:
            ax.imshow(results[key][18], vmin=min_vals[key], vmax=max_vals[key], cmap=cmap)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        fig.add_subplot(ax)
        norm = mpl.colors.Normalize(vmin=min_vals[key], vmax=max_vals[key])
        cax = inset_axes(
            ax,
            width="100%",
            height="10%",
            bbox_to_anchor=(0, -0.2, 1, 1),
            bbox_transform=ax.transAxes,
            loc="lower center",
        )
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
        cb1.outline.set_color('black')
    name = f'visuals/hist_param_amc_{case}.png'
    plt.savefig(name, dpi=150, bbox_inches='tight')
    plt.show()


def lines_box_plot_amc(method='PPINN'):
    if method == 'PPINN':
        base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    elif method == 'SPPINN':
        base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_sppinn_amcctp"
    elif method == 'SV':
        base = r"D:\PPINN_patient_data\sygnovia"
    else:
        raise NotImplementedError('Method not implemented.')
    cases = [x for x in os.listdir(base) if 'C1' in x]
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}
    min_vals = {'cbf': 0.0, 'mtt': 0.0, 'cbv': 0., 'delay': 0.0, 'tmax': 0.0, 'dwi_seg': 0.0}
    max_vals = {'cbf': 120, 'mtt': 15, 'cbv': 5, 'delay': 9, 'tmax': 15, 'dwi_seg': 1}
    fig = plt.figure(figsize=(15, 3))
    outer = gridspec.GridSpec(1, 5, wspace=0.55, hspace=0.55)
    for i, key in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
        infarcted = []
        healthy = []
        volumes = []
        for case in cases:
            if method == 'SV':
                results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii.gz")),
                           'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii.gz")),
                           'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii.gz")),
                           'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii.gz")),
                           'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii.gz")),
                           'dwi_seg': sitk.ReadImage(
                               os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                            "DWI_seg_registered_corrected.nii.gz")),
                           'contra': sitk.ReadImage(
                               os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                            "DWI_seg_registered_contralateral.nii.gz"))
                           }

            else:
                results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                           'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                           'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                           'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                           'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                           'vesselmask': sitk.ReadImage(os.path.join(base, case, "vesselmask.nii")),
                           'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                           'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz")),
                           'contra': sitk.ReadImage(
                               os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                            "DWI_seg_registered_contralateral.nii.gz"))
                           }



            spacing = results['dwi_seg'].GetSpacing()
            results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}
            results = drop_unphysical_per_method(results)

            mtt = sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\sygnovia", case, "mtt.nii.gz"))
            mtt = sitk.GetArrayFromImage(mtt)
            vesselmask = np.zeros_like(mtt)
            vesselmask[mtt >= 0] = 1
            results['vesselmask'] = vesselmask

            volume = np.round(np.sum(results['dwi_seg']) * np.product(spacing) / 1000, 1)
            volumes.append(volume)

            results['healthy'] = results['contra'] * results['vesselmask']
            results['infarcted'] = results['dwi_seg'] * results['vesselmask']
            healthy.append(np.mean(results[key][results['healthy'] == 1].flatten()))
            infarcted.append(np.mean(results[key][results['infarcted'] == 1].flatten()))
            if key in ['mtt', 'tmax', 'delay']:
                if np.mean(results[key][results['infarcted'] == 1].flatten()) < np.mean(
                        results[key][results['healthy'] == 1].flatten()):
                    print(case, np.mean(results[key][results['infarcted'] == 1].flatten()) - np.mean(
                        results[key][results['healthy'] == 1].flatten()))
        ax = plt.Subplot(fig, outer[i])
        ax.set_title(f'{param_title[key]}', size=14, pad=10)
        ax.boxplot([healthy, infarcted], widths=(0.3, 0.3), medianprops=dict(linestyle='-', linewidth=2., color='k'))
        ax.scatter(15 * [1], healthy, alpha=0.5, c='green')
        ax.scatter(15 * [2], infarcted, alpha=0.5, c='firebrick')
        ax.set_ylabel(param_unts[key], fontdict=font)
        ax.set_ylim(min_vals[key], max_vals[key])

        for i, j in zip(healthy, infarcted):
            if key in ['mtt', 'delay', 'tmax']:
                if j < i * 0.98:
                    ax.plot([1, 2], [i, j], c='firebrick', alpha=0.5)
                elif j > i * 1.02:
                    ax.plot([1, 2], [i, j], c='green', alpha=0.5)
                else:
                    ax.plot([1, 2], [i, j], c='k', alpha=0.5)
            if key in ['cbf', 'cbv']:
                if j < i * 0.98:
                    ax.plot([1, 2], [i, j], c='green', alpha=0.5)
                elif j > i * 1.02:
                    ax.plot([1, 2], [i, j], c='firebrick', alpha=0.5)
                else:
                    ax.plot([1, 2], [i, j], c='k', alpha=0.5)
        labels = ['Healthy', 'Infarcted']
        ax.set_xticklabels(labels, fontdict=font)
        fig.add_subplot(ax)
    plt.savefig(f'visuals/lines_boxplot_contralateral_{method}.png', dpi=150, bbox_inches='tight')

    plt.show()


def plot_sv_vs_ppinn():
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    cases = [x for x in os.listdir(base) if 'C1' in x]

    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 0, 'tmax': 0}
    max_vals = {'cbf': 100, 'mtt': 28, 'cbv': 6, 'delay': 3.5, 'tmax': 20}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}
    for case in tqdm(cases):
        results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                   'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                   'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                   'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                   'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                   'vesselmask': sitk.ReadImage(os.path.join(base, case, "vesselmask.nii")),
                   'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                   'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz"))}
        results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}
        results_sv = load_sygnovia_results_amc(case)
        svmask = results_sv['cbf'] + results_sv['mtt'] + results_sv['cbv'] + results_sv['delay'] + results_sv['tmax']
        results_sv['sv_mask'] = np.zeros_like(results_sv['cbf'])
        results_sv['sv_mask'][svmask != 0] = 1
        min_vals_ppinn = {'cbf': np.percentile(results['cbf'][results['vesselmask'] == 1], q=10),
                          'mtt': np.percentile(results['mtt'][results['vesselmask'] == 1], q=10),
                          'cbv': np.percentile(results['cbv'][results['vesselmask'] == 1], q=10),
                          'delay': np.percentile(results['delay'][results['vesselmask'] == 1], q=10),
                          'tmax': np.percentile(results['tmax'][results['vesselmask'] == 1], q=10)}

        max_vals_ppinn = {'cbf': np.percentile(results['cbf'][results['vesselmask'] == 1], q=90),
                          'mtt': np.percentile(results['mtt'][results['vesselmask'] == 1], q=90),
                          'cbv': np.percentile(results['cbv'][results['vesselmask'] == 1], q=90),
                          'delay': np.percentile(results['delay'][results['vesselmask'] == 1], q=90),
                          'tmax': np.percentile(results['tmax'][results['vesselmask'] == 1], q=90)}

        min_vals_sv = {'cbf': np.percentile(results_sv['cbf'][results_sv['sv_mask'] == 1], q=10),
                       'mtt': np.percentile(results_sv['mtt'][results_sv['sv_mask'] == 1], q=10),
                       'cbv': np.percentile(results_sv['cbv'][results_sv['sv_mask'] == 1], q=10),
                       'delay': np.percentile(results_sv['delay'][results_sv['sv_mask'] == 1], q=10),
                       'tmax': np.percentile(results_sv['tmax'][results_sv['sv_mask'] == 1], q=10)}

        max_vals_sv = {'cbf': np.percentile(results_sv['cbf'][results_sv['sv_mask'] == 1], q=90),
                       'mtt': np.percentile(results_sv['mtt'][results_sv['sv_mask'] == 1], q=90),
                       'cbv': np.percentile(results_sv['cbv'][results_sv['sv_mask'] == 1], q=90),
                       'delay': np.percentile(results_sv['delay'][results_sv['sv_mask'] == 1], q=90),
                       'tmax': np.percentile(results_sv['tmax'][results_sv['sv_mask'] == 1], q=90)}

        for k in range(results_sv['core'].shape[0]):
            fig = plt.figure(figsize=(9, 4))
            outer = gridspec.GridSpec(1, 5, wspace=0.2, hspace=0.2)
            for i, key in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
                inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                         subplot_spec=outer[i],
                                                         wspace=0.1,
                                                         hspace=0.1)
                ax = plt.Subplot(fig, inner[0])
                ax.imshow(results[key][k], cmap='jet', vmin=min_vals_ppinn[key], vmax=max_vals_ppinn[key])
                ax.set_title(param_title[key], fontdict=font)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_axis_off()
                if i == 0:
                    ax.set_ylabel('PPINN', fontdict=font)
                fig.add_subplot(ax)
                norm = mpl.colors.Normalize(vmin=min_vals_ppinn[key], vmax=max_vals_ppinn[key])
                cax = inset_axes(
                    ax,
                    width="100%",
                    height="10%",
                    bbox_to_anchor=(0, -0.2, 1, 1),
                    bbox_transform=ax.transAxes,
                    loc="lower center",
                )
                cb1 = mpl.colorbar.ColorbarBase(cax, cmap='jet', norm=norm, orientation='horizontal')
                cb1.outline.set_color('black')

                ax = plt.Subplot(fig, inner[1])
                ax.imshow(results_sv[key][k], cmap='jet', vmin=min_vals_sv[key], vmax=max_vals_sv[key])
                # ax.set_title(param_title[key], fontdict=font)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_axis_off()
                if i == 0:
                    ax.set_ylabel('Sygno.via', fontdict=font)
                fig.add_subplot(ax)
                norm = mpl.colors.Normalize(vmin=min_vals_sv[key], vmax=max_vals_sv[key])
                cax = inset_axes(
                    ax,
                    width="100%",
                    height="10%",
                    bbox_to_anchor=(0, -0.2, 1, 1),
                    bbox_transform=ax.transAxes,
                    loc="lower center",
                )
                cb1 = mpl.colorbar.ColorbarBase(cax, cmap='jet', norm=norm, orientation='horizontal')
                cb1.outline.set_color('black')
            if np.sum(results['dwi_seg'][k]) > 0:
                name = f'visuals/compare/pt_{str(case)}_slice_{str(k)}.png'
                plt.savefig(name, dpi=100, bbox_inches='tight')
                plt.close()
            else:
                plt.close()


def plot_ppinn_maps(case, slice):
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    min_vals = {'scan':0.1, 'cbf': 0.01, 'mtt':2 , 'cbv': 0.01, 'delay': 0.01, 'tmax': 2}
    max_vals = {'scan':150, 'cbf': 35, 'mtt': 8, 'cbv': 3, 'delay': 2., 'tmax': 6}
    param_title = {'scan':'scan','cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}
    results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
               'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
               'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
               'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
               'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
               'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
               'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz")),
               'scan': sitk.ReadImage(rf"D:\PPINN_patient_data\AMCCTP\CTP_nii_registered\{case}\14.nii.gz")}
    results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}
    cmap = mpl.cm.get_cmap('jet').copy()
    cmap.set_under(color='black')
    cmap_bw = mpl.cm.get_cmap('Greys_r').copy()
    cmap_bw.set_under(color='black')
    for k in range(results['dwi_seg'].shape[0]):
        if k == slice:
            for i, key in zip(range(6), ['scan','cbf', 'mtt', 'cbv', 'delay', 'tmax']):
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                if key == 'scan':
                    ax.imshow(results[key][k], cmap=cmap_bw, vmin=min_vals[key], vmax=max_vals[key])
                else:
                    ax.imshow(results[key][k], cmap=cmap, vmin=min_vals[key], vmax=max_vals[key])

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_axis_off()
                name = f'visuals/param_{key}_case_{str(case)}_slice_{str(slice)}.png'
                plt.savefig(name, dpi=150, bbox_inches='tight')
                plt.show()

if __name__ == '__main__':
    # lines_box_plot_amc(method='SV')
    lines_box_plot_amc(method='SPPINN')