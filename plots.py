import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np
import os
from utils.val_utils import drop_edges, drop_unphysical
from utils.val_utils import load_sygnovia_results, load_nlr_results, load_phantom_gt, load_sygnovia_results_amc
from utils.plot_utils import bland_altman_plot
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SimpleITK as sitk
from einops.einops import rearrange, repeat

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 12,
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

# plt.rcParams['text.usetex'] = True
def load_phantom(folder=r'data/DigitalPhantomCT', cbv_ml=5, simulation_method=2):
    print("Reading Dicom directory:", folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_data = sitk.GetArrayFromImage(image)
    # values: AIF/VOF, Exp R(t) for CBV 1-5, Lin R(t) for CBV 1-5, Box R(t) for CBV 1-5,
    image_data = rearrange(image_data, '(t values) h w -> values t h w', t=30)
    image_data = image_data.astype(np.float32)
    time = np.array([float(x) for x in range(0, 60, 2)])
    vof_location = (410, 247, 16)  # start, start, size
    vof_data = image_data[0,
               :,
               vof_location[0]:vof_location[0] + vof_location[2],
               vof_location[1]:vof_location[1] + vof_location[2]]
    vof_data = np.mean(vof_data, axis=(1, 2))

    aif_location = (123, 251, 8)  # start, start, size
    aif_data = image_data[0,
               :,
               aif_location[0]:aif_location[0] + aif_location[2],
               aif_location[1]:aif_location[1] + aif_location[2]]
    aif_data = np.mean(aif_data, axis=(1, 2))
    simulated_data_size = 32 * 7
    scan_center = image_data.shape[-1] // 2
    simulated_data_start = scan_center - simulated_data_size // 2
    simulated_data_end = scan_center + simulated_data_size // 2
    perfusion_data = image_data[1:, :,
                     simulated_data_start:simulated_data_end,
                     simulated_data_start:simulated_data_end]
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
    perfusion_values_dict = {'cbf': perfusion_values[simulation_method, cbv_ml - 1, ..., 3],
                             'delay': perfusion_values[simulation_method, cbv_ml - 1, ..., 1],
                             'cbv': perfusion_values[simulation_method, cbv_ml - 1, ..., 0],
                             'mtt': perfusion_values[simulation_method, cbv_ml - 1, ..., 2] * 60}
    perfusion_values_dict['tmax'] = perfusion_values_dict['delay'] + 0.5 * perfusion_values_dict['mtt']
    perfusion_data = rearrange(perfusion_data, '(n cbv) t h w -> n cbv h w t', n=3)
    return {'time': time,
            'vof': vof_data,
            'aif': aif_data,
            'perfusion_data': perfusion_data,
            'perfusion_values': perfusion_values_dict
            }


def load_ppinn_results(cbv_ml=5, sd=2, undersample=False):
    us = 0.5 if undersample else 0
    with open(
            rf'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\ppinn_results\ppinn_results_cbv_{cbv_ml}_sd_{sd}_undersample_{us}.pickle',
            'rb') as f:
        results = pickle.load(f)
    return results


def get_data(result_dict_function, sd, undersample=False):
    if not sd:
        data = result_dict_function(cbv_ml=1)
        for key in data.keys():
            data[key] = np.concatenate(
                [result_dict_function(cbv_ml=ml, undersample=undersample)[key] for ml in [1, 2, 3, 4, 5]])
        return data
    else:
        data = result_dict_function(cbv_ml=1, sd=sd)
        for key in data.keys():
            data[key] = np.concatenate(
                [result_dict_function(cbv_ml=ml, sd=sd, undersample=undersample)[key] for ml in [1, 2, 3, 4, 5]])
        return data


def make_axes(ax):
    x1 = [112 + 224 * x for x in [0, 1, 2, 3, 4]]
    names = ['1 ml ', '2 ml ', '3 ml ', '4 ml ', '5 ml ']
    ax[0].set_title('GT', fontdict=font)
    ax[1].set_title('Sygno.via', fontdict=font)
    ax[2].set_title('NLR', fontdict=font)
    for x in ax.flatten():
        x.tick_params(axis=u'both', which=u'both', length=0)
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    ax[0].set_yticks(x1)
    ax[0].set_yticklabels(names, minor=False, fontdict=font)
    ax[0].set_ylabel('CBV', fontdict=font)


def make_grid_plot(sd, drop=False, undersample=False):
    results = {'gt': get_data(load_phantom_gt, sd=None, undersample=False),
               'sygnovia': get_data(load_sygnovia_results, sd=sd, undersample=undersample),
               'nlr': get_data(load_nlr_results, sd=sd, undersample=undersample),
               'ppinn': get_data(load_ppinn_results, sd=sd, undersample=undersample)}
    if drop:
        results = drop_edges(results)
        results = drop_unphysical(results)

    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 0, 'tmax': 0}
    max_vals = {'cbf': 100, 'mtt': 28, 'cbv': 6, 'delay': 3.5, 'tmax': 20}
    title = {'gt': 'GT', 'sygnovia': 'Sygno.via', 'nlr': 'NLR', 'ppinn': 'PPINN'}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}

    x1 = [112 + 224 * x for x in [0, 1, 2, 3, 4]]
    if drop:
        x1 = [110 + 220 * x for x in [0, 1, 2, 3, 4]]
    names = ['1 ml ', '2 ml ', '3 ml ', '4 ml ', '5 ml ']

    fig = plt.figure(figsize=(10, 15))
    outer = gridspec.GridSpec(3, 2, wspace=0.55, hspace=0.3)
    for i, param in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
        inner = gridspec.GridSpecFromSubplotSpec(1, 4,
                                                 subplot_spec=outer[i],
                                                 wspace=0.1,
                                                 hspace=0.1)
        ax = plt.Subplot(fig, outer[i])
        if i != 5:
            ax.set_title(f'{param_title[param]}', size=14, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        fig.add_subplot(ax)
        for j, key in zip(range(5), ['gt', 'sygnovia', 'nlr', 'ppinn', '']):
            if j < 4:
                ax = plt.Subplot(fig, inner[j])
                ax.imshow(results[key][param], cmap='jet', vmin=min_vals[param], vmax=max_vals[param])
                ax.set_title(title[key], fontdict=font)
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0 and (i == 0 or i == 3):
                    ax.tick_params(axis=u'both', which=u'both', length=0)
                    ax.set_yticks(x1)
                    ax.set_yticklabels(names, minor=False, fontdict=font)
                    ax.set_ylabel('CBV', fontdict=font)

                fig.add_subplot(ax)
            else:
                norm = mpl.colors.Normalize(vmin=min_vals[param], vmax=max_vals[param])
                cax = inset_axes(
                    ax,
                    width="20%",
                    height="100%",
                    bbox_to_anchor=(0.5, 0, 1, 1),
                    bbox_transform=ax.transAxes,
                    loc="right",
                )
                cb1 = mpl.colorbar.ColorbarBase(cax, cmap='jet', norm=norm, orientation='vertical')
                cb1.outline.set_color('black')
                cb1.set_label(param_unts[param], fontdict=font)

            if j == 0:
                ax.tick_params(axis=u'both', which=u'both', length=0)
                ax.set_yticks(x1)
                ax.set_yticklabels(names, minor=False, fontdict=font)
                ax.set_ylabel('CBV', fontdict=font)

            fig.add_subplot(ax)

    name = f'visuals/phantom_results_sd_{sd}_us_{str(undersample)}_drop_{str(drop)}.png'
    plt.savefig(name, dpi=150, bbox_inches='tight')
    fig.show()


def make_ba_plots(sd, drop=False, undersample=False):
    results = {'gt': get_data(load_phantom_gt, sd=None, undersample=False),
               'sygnovia': get_data(load_sygnovia_results, sd=sd, undersample=undersample),
               'nlr': get_data(load_nlr_results, sd=sd, undersample=undersample),
               'ppinn': get_data(load_ppinn_results, sd=sd, undersample=undersample)}
    if drop:
        results = drop_edges(results)
        results = drop_unphysical(results)

    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 0, 'tmax': 0}
    max_vals = {'cbf': 100, 'mtt': 28, 'cbv': 6, 'delay': 3.5, 'tmax': 20}

    title = {'gt': 'GT', 'sygnovia': 'Sygno.via', 'nlr': 'NLR', 'ppinn': 'PPINN'}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}

    for key in param_title.keys():
        fig, ax = plt.subplots(1, 3)
        scale = 1
        bland_altman_plot(results['sygnovia'][key].flatten(), results['gt'][key].flatten(), ax[0],
                          xlim=max_vals[key] * scale, ylim=max_vals[key] * scale, s=0.1)
        bland_altman_plot(results['nlr'][key].flatten(), results['gt'][key].flatten(), ax[1],
                          xlim=max_vals[key] * scale, ylim=max_vals[key] * scale, s=0.1)
        bland_altman_plot(results['ppinn'][key].flatten(), results['gt'][key].flatten(), ax[2],
                          xlim=max_vals[key] * scale, ylim=max_vals[key] * scale, s=0.1)
        ax[0].set_title('Sygno.via')
        ax[1].set_title('NLR')
        ax[2].set_title('PPINN')
        fig.suptitle(param_title[key] + f',sd: {sd}, unphysical dropped: {drop}', fontsize=16)
        name = f'visuals/boxplots_param_{key}_sd_{sd}_us_{str(undersample)}_drop_{str(drop)}.png'
        plt.savefig(name, dpi=150, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    # for key in param_title.keys():
    #     fig, ax = plt.subplots(1,3)
    #     fig.suptitle(param_title[key], fontsize=16)
    #     ax[0].scatter(results['gt'][key].flatten(), results['sygnovia'][key].flatten(), s=0.2)
    #     ax[1].scatter(results['gt'][key].flatten(), results['nlr'][key].flatten(), s=0.2)
    #     ax[2].scatter(results['gt'][key].flatten(), results['ppinn'][key].flatten(), s=0.2)
    #     for x in ax:
    #         x.set_ylim(0,max_vals[key]*1.2)
    #         x.set_xlim(0,max_vals[key]*1.2)
    #     plt.show()


def make_amc_param_plots(contralateral=False):
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    cases = os.listdir(base)[:-1]
    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 0, 'tmax': 0}
    max_vals = {'cbf': 100, 'mtt': 28, 'cbv': 6, 'delay': 3.5, 'tmax': 20}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    for case in cases:
        results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                   'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                   'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                   'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                   'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                   'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                   'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz")),
                   'contra': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                         "DWI_seg_registered_contralateral.nii.gz"))
                   }
        spacing = results['dwi_seg'].GetSpacing()
        results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}
        volume = np.round(np.sum(results['dwi_seg']) * np.product(spacing) / 1000, 1)

        if not contralateral:
            results['dwi_seg_inverse'] = np.zeros_like(results['dwi_seg'])
            results['dwi_seg_inverse'][results['dwi_seg'] != 1] = 1
            results['healthy'] = results['dwi_seg_inverse'] * results['brainmask']
        if contralateral:
            results['dwi_seg_inverse'] = results['contra']
            results['healthy'] = results['dwi_seg_inverse'] * results['brainmask']

        fig = plt.figure(figsize=(10, 5))
        outer = gridspec.GridSpec(2, 3, wspace=0.55, hspace=0.55)
        for i, key in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
            kwargs = {'bins': 100, 'density': True, 'histtype': 'step', 'range': (min_vals[key], max_vals[key] * 2)}
            ax = plt.Subplot(fig, outer[i])
            ax.set_title(f'{param_title[key]}', size=14, pad=20)
            ax.hist(results[key][results['healthy'] == 1].flatten(), label='healthy', **kwargs)
            ax.hist(results[key][results['dwi_seg'] == 1].flatten(), label=f'infarcted: {volume} ml', **kwargs)
            if i == 4:
                ax.legend(title=case, loc='center left', bbox_to_anchor=(1, 0.5))

            fig.add_subplot(ax)
        name = f'visuals/hist_param_amc_{case}.png'
        plt.savefig(name, dpi=150, bbox_inches='tight')
        plt.show()


def lines_box_plot_amc(contralateral=False):
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    cases = os.listdir(base)[:-1]
    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 0, 'tmax': 0}
    max_vals = {'cbf': 100, 'mtt': 28, 'cbv': 6, 'delay': 3.5, 'tmax': 20}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}

    fig = plt.figure(figsize=(15, 3))
    outer = gridspec.GridSpec(1, 5, wspace=0.55, hspace=0.55)
    for i, key in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
        infarcted = []
        healthy = []
        volumes = []
        for case in cases:
            results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                       'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                       'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                       'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                       'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                       'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                       'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz")),
                       'mip': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MIP", case, "mip.nii.gz")),
                       'contra': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                             "DWI_seg_registered_contralateral.nii.gz"))}
            # TODO
            # here we should remove vessels
            spacing = results['dwi_seg'].GetSpacing()
            results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}

            # TODO
            # here we should remove vessels

            volume = np.round(np.sum(results['dwi_seg']) * np.product(spacing) / 1000, 1)
            volumes.append(volume)
            if not contralateral:
                results['dwi_seg_inverse'] = np.zeros_like(results['dwi_seg'])
                results['dwi_seg_inverse'][results['dwi_seg'] != 1] = 1
                results['healthy'] = results['dwi_seg_inverse'] * results['brainmask']
                healthy.append(np.mean(results[key][results['healthy'] == 1].flatten()))
                infarcted.append(np.mean(results[key][results['dwi_seg'] == 1].flatten()))
            if contralateral:
                results['dwi_seg_inverse'] = results['contra']
                results['healthy'] = results['dwi_seg_inverse'] * results['brainmask']
                healthy.append(np.mean(results[key][results['healthy'] == 1].flatten()))
                infarcted.append(np.mean(results[key][results['dwi_seg'] == 1].flatten()))
            if key in ['mtt', 'tmax', 'delay']:
                if np.mean(results[key][results['dwi_seg'] == 1].flatten()) < np.mean(
                        results[key][results['healthy'] == 1].flatten()):
                    print(case, np.mean(results[key][results['dwi_seg'] == 1].flatten()) - np.mean(
                        results[key][results['healthy'] == 1].flatten()))

        ax = plt.Subplot(fig, outer[i])
        ax.set_title(f'{param_title[key]}', size=14, pad=10)
        ax.boxplot([healthy, infarcted], widths=(0.3, 0.3), medianprops=dict(linestyle='-', linewidth=2., color='k'))
        ax.scatter(15 * [1], healthy, alpha=0.5, c='green')
        ax.scatter(15 * [2], infarcted, alpha=0.5, c='firebrick')
        ax.set_ylabel(param_unts[key], fontdict=font)

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
    plt.savefig(f'visuals/lines_boxplot_contra_{contralateral}.png', dpi=150, bbox_inches='tight')

    plt.show()


def plot_sv_vs_ppinn():
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    cases = os.listdir(base)[:-1]

    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 0, 'tmax': 0}
    max_vals = {'cbf': 100, 'mtt': 28, 'cbv': 6, 'delay': 3.5, 'tmax': 20}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}
    for case in cases:
        results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                   'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                   'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                   'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                   'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                   'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                   'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz"))}
        results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}

        results_sv = load_sygnovia_results_amc(case)
        for k in range(results_sv['core'].shape[0]):
            fig = plt.figure(figsize=(4, 8))

            outer = gridspec.GridSpec(5, 1, wspace=0.55, hspace=0.55)
            for i, key in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
                inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                         subplot_spec=outer[i],
                                                         wspace=0.1,
                                                         hspace=0.1)
                ax = plt.Subplot(fig, inner[0])
                ax.imshow(results[key][k], cmap='jet', vmin=min_vals[key], vmax=max_vals[key])
                ax.set_title('PPINN' + param_title[key], fontdict=font)
                fig.add_subplot(ax)
                ax = plt.Subplot(fig, inner[1])
                ax.imshow(results_sv[key][k], cmap='jet', vmin=min_vals[key], vmax=max_vals[key])
                ax.set_title('SV' + param_title[key], fontdict=font)
                fig.add_subplot(ax)
        if np.sum(results['dwi_seg']) > 0:
            plt.show()
        else:
            plt.close()


def plot_ppinn_maps(case, slice):
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 1, 'tmax': 0}
    max_vals = {'cbf': 35, 'mtt': 13, 'cbv': 3, 'delay': 2.5, 'tmax': 12}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    param_unts = {'cbf': '[ml/100g/min]', 'mtt': '[s]', 'cbv': '[ml/100g]', 'delay': '[s]', 'tmax': '[s]'}
    results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
               'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
               'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
               'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
               'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
               'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
               'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz"))}
    results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}

    for k in range(results['dwi_seg'].shape[0]):
        if k == slice:
            for i, key in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                ax.imshow(results[key][k], cmap='jet', vmin=min_vals[key], vmax=max_vals[key])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_axis_off()
                name = f'visuals/param_{key}_case_{str(case)}_slice_{str(slice)}.png'
                plt.savefig(name, dpi=150, bbox_inches='tight')
                plt.show()


def make_phantom_plot():
    data = load_phantom()
    xs = [15, 200, 150]
    ys = [222, 20, 90]
    col = ['k', 'slategrey', 'silver']

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    ax[0].plot(data['time'], data['aif'], label=r'$C_{AIF}(t)$', c='firebrick', lw=2)
    ax[0].plot(data['time'], data['vof'], label=r'$C_{VOF}(t)$', c='royalblue', lw=2)

    for ix, (i, j) in enumerate(zip(xs, ys)):
        cbv = np.round(data['perfusion_values']['cbv'][i, j], 1)
        mtt = np.round(data['perfusion_values']['mtt'][i, j], 1)
        delay = np.round(data['perfusion_values']['delay'][i, j], 1)
        tac = data['perfusion_data'][2, 4, i, j, :]
        ax[1].plot(data['time'], tac, label=f'CBV: {cbv} ml/100g,\nMTT: {mtt} s, $t_d$: {delay} s', c=col[ix], lw=2)

    ax[1].set_ylim(20, 100)
    ax[0].set_ylim(20, 300)

    # ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].legend(frameon=False, prop={'size': 12})
    ax[1].legend(frameon=False, title=r'$C_{TAC}(t)$', prop={'size': 10})

    ax[1].set_xlabel(r'[s]', fontdict=font)
    ax[0].set_xlabel(r'[s]', fontdict=font)
    ax[0].set_ylabel(r'[HU]', fontdict=font)
    # ax[1].set_ylabel(r'[HU]', fontdict=font)
    for x_ in ax:
        x_.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('visuals/phantom_cruves.png', dpi=150, bbox_inches='tight')

    plt.show()

    print('hi')


if __name__ == '__main__':
    # make_ba_plots(sd=2, drop=True)
    # make_ba_plots(sd=2, drop=False)
    # make_plots(sd=4, drop=True)
    # make_grid_plot(sd=1, drop=True, undersample=False)
    # make_grid_plot(sd=2, drop=True, undersample=False)
    # make_grid_plot(sd=3, drop=True, undersample=False)
    # make_grid_plot(sd=4, drop=True, undersample=False)
    # make_grid_plot(sd=5, drop=True, undersample=False)
    # make_grid_plot(sd=2, drop=True, undersample=True)
    # make_grid_plot(sd=2, drop=False, undersample=False)
    # make_grid_plot(sd=3, drop=False, undersample=False)
    # make_grid_plot(sd=4, drop=False, undersample=False)
    # make_grid_plot(sd=5, drop=False, undersample=False)
    # make_grid_plot(sd=2, drop=False, undersample=True)
    make_amc_param_plots(contralateral=False)
    # lines_box_plot_amc(contralateral=True)
    # lines_box_plot_amc(contralateral=False)

    # plot_sv_vs_ppinn()
    # plot_ppinn_maps('C114', slice=11)
