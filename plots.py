import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np
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
# plt.rcParams['text.usetex'] = True
def load_phantom(folder=r'data/DigitalPhantomCT',cbv_ml=5, simulation_method=2):
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
    perfusion_values_dict = {'cbf': perfusion_values[simulation_method, cbv_ml-1, ..., 3],
                             'delay': perfusion_values[simulation_method, cbv_ml-1, ..., 1],
                             'cbv': perfusion_values[simulation_method, cbv_ml-1, ..., 0],
                             'mtt': perfusion_values[simulation_method, cbv_ml-1, ..., 2] * 60}
    perfusion_values_dict['tmax'] = perfusion_values_dict['delay'] + 0.5 * perfusion_values_dict['mtt']
    perfusion_data = rearrange(perfusion_data, '(n cbv) t h w -> n cbv h w t', n=3)
    return {'time': time,
            'vof': vof_data,
            'aif': aif_data,
            'perfusion_data':perfusion_data,
            'perfusion_values':perfusion_values_dict
            }

def load_ppinn_results(cbv_ml=5, sd=2, undersample=False):
    with open(rf'C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\ppinn_results\ppinn_results_cbv_{cbv_ml}_sd_{sd}.pickle', 'rb') as f:
        results = pickle.load(f)
    return results

def get_data(result_dict_function, sd):
    if not sd:
        data = result_dict_function(cbv_ml=1)
        for key in data.keys():
            data[key] = np.concatenate([result_dict_function(cbv_ml=ml)[key] for ml in [1,2,3,4,5]])
        return data
    else:
        data = result_dict_function(cbv_ml=1, sd=sd)
        for key in data.keys():
            data[key] = np.concatenate([result_dict_function(cbv_ml=ml, sd=sd)[key] for ml in [1,2,3,4,5]])
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

def make_grid_plot(sd, drop=False):
    results = {'gt': get_data(load_phantom_gt, sd=None),
               'sygnovia': get_data(load_sygnovia_results, sd=2),
               'nlr': get_data(load_nlr_results, sd=2),
               'ppinn': get_data(load_ppinn_results, sd=2)}
    if drop:
        results = drop_edges(results)
        results = drop_unphysical(results)

    min_vals = {'cbf':0, 'mtt':0, 'cbv':0, 'delay':0, 'tmax':0}
    max_vals = {'cbf':100, 'mtt':28, 'cbv':6, 'delay':3.5, 'tmax':15}
    title = {'gt': 'GT', 'sygnovia':'Sygno.via', 'nlr':'NLR', 'ppinn':'PPINN'}
    param_title = {'cbf':'CBF', 'mtt':'MTT', 'cbv':'CBV', 'delay':'Delay', 'tmax':'Tmax'}
    param_unts = {'cbf':'[ml/100g/min]', 'mtt':'[s]', 'cbv':'[ml/100g]', 'delay':'[s]', 'tmax':'[s]'}

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
        for j,key in zip(range(5), ['gt', 'sygnovia', 'nlr', 'ppinn', '']):
            if j<4:
                ax = plt.Subplot(fig, inner[j])
                ax.imshow(results[key][param], cmap='jet', vmin=min_vals[param], vmax=max_vals[param])
                ax.set_title(title[key], fontdict=font)
                ax.set_xticks([])
                ax.set_yticks([])
                if j==0 and (i==0 or i == 3):
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

    plt.savefig('phantom_results.png', dpi=150, bbox_inches='tight')
    fig.show()
def make_plots(sd, drop=False):
    results = {'gt': get_data(load_phantom_gt, sd=None),
               'sygnovia': get_data(load_sygnovia_results, sd=2),
               'nlr': get_data(load_nlr_results, sd=2)}
    if drop:
        results = drop_edges(results)
        results = drop_unphysical(results)


    fig, ax = plt.subplots(1,3, figsize=(6,10))
    ax[0].imshow(results['gt']['cbf'], cmap='jet', vmin=0, vmax=90)
    ax[1].imshow(results['sygnovia']['cbf'], cmap='jet', vmin=0, vmax=90)
    ax[2].imshow(results['nlr']['cbf'], cmap='jet', vmin=0, vmax=90)
    make_axes(ax)
    plt.subplots_adjust(left=0.3)
    plt.show()

    fig, ax = plt.subplots(1,3, figsize=(6,10))
    ax[0].imshow(results['gt']['cbv'], cmap='jet', vmin=0, vmax=7)
    ax[1].imshow(results['sygnovia']['cbv'], cmap='jet', vmin=0, vmax=7)
    ax[2].imshow(results['nlr']['cbv'], cmap='jet', vmin=0, vmax=7)
    make_axes(ax)
    plt.subplots_adjust(left=0.3)

    plt.show()
    fig, ax = plt.subplots(1,3, figsize=(6,10))
    ax[0].imshow(results['gt']['mtt'], cmap='jet', vmin=0, vmax=25)
    ax[1].imshow(results['sygnovia']['mtt'], cmap='jet', vmin=0, vmax=25)
    ax[2].imshow(results['nlr']['mtt'], cmap='jet', vmin=0, vmax=25)
    make_axes(ax)
    plt.subplots_adjust(left=0.3)

    plt.show()
    fig, ax = plt.subplots(1,3, figsize=(6,10))
    ax[0].imshow(results['gt']['tmax'], cmap='jet', vmin=0, vmax=15)
    ax[1].imshow(results['sygnovia']['tmax'], cmap='jet', vmin=0, vmax=15)
    ax[2].imshow(results['nlr']['tmax'], cmap='jet', vmin=0, vmax=15)
    make_axes(ax)
    plt.subplots_adjust(left=0.3)

    plt.show()
    ffig, ax = plt.subplots(1,3, figsize=(6,10))
    ax[0].imshow(results['gt']['delay'], cmap='jet', vmin=0, vmax=3.5)
    ax[1].imshow(results['sygnovia']['delay'], cmap='jet', vmin=0, vmax=3.5)
    ax[2].imshow(results['nlr']['delay'], cmap='jet', vmin=0, vmax=3.5)
    make_axes(ax)
    plt.subplots_adjust(left=0.3)

    plt.show()


    fig, ax = plt.subplots(1,3)
    bland_altman_plot(results['sygnovia']['cbv'].flatten(), results['nlr']['cbv'].flatten(), ax[0], xlim=10, ylim=10, s=0.2)
    bland_altman_plot(results['sygnovia']['cbv'].flatten(), results['gt']['cbv'].flatten(), ax[1], xlim=10, ylim=10, s=0.2)
    bland_altman_plot(results['nlr']['cbv'].flatten(), results['gt']['cbv'].flatten(), ax[2], xlim=10, ylim=10, s=0.2)
    plt.show()

    fig, ax = plt.subplots(1,3)
    ax[1].scatter(results['gt']['cbf'].flatten(), results['sygnovia']['cbf'].flatten(), s=0.2)
    ax[2].scatter(results['gt']['cbf'].flatten(), results['nlr']['cbf'].flatten(), s=0.2)

    ax[1].set_ylim(0,150)
    ax[2].set_ylim(0,150)
    ax[1].set_xlim(0,100)
    ax[2].set_xlim(0,100)
    plt.show()

def make_phantom_plot():
    data = load_phantom()
    xs = [15, 200, 150]
    ys = [222, 20, 90]
    col = ['k', 'slategrey', 'orange']

    fig = plt.figure()
    ax = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)]
    ax[0].plot(data['time'], data['aif'], label=r'$C_{AIF}(t)$', c='royalblue', lw=2)
    ax[0].plot(data['time'], data['vof'], label=r'$C_{VOF}(t)$', c='firebrick', lw=2)

    for ix, (i, j )in enumerate(zip(xs, ys)):
        cbv = np.round(data['perfusion_values']['cbv'][i,j],1)
        mtt = np.round(data['perfusion_values']['mtt'][i,j],1)
        delay = np.round(data['perfusion_values']['delay'][i,j],1)
        tac = data['perfusion_data'][2,4,i,j,:]
        ax[1].plot(data['time'], tac, label=f'CBV: {cbv} ml/100g,\nMTT: {mtt} s, $t_d$: {delay} s',c=col[ix], lw=2)

    ax[1].set_ylim(20,100)
    ax[0].set_ylim(20,300)

    # ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].legend(frameon=False)
    ax[1].legend(frameon=False)

    ax[1].set_xlabel(r'[s]', fontdict=font)
    ax[0].set_xlabel(r'[s]', fontdict=font)
    ax[0].set_ylabel(r'[HU]', fontdict=font)
    # ax[1].set_ylabel(r'[HU]', fontdict=font)
    for x_ in ax:
        x_.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig('phantom_cruves.png', dpi=150, bbox_inches='tight')

    plt.show()

    print('hi')


if __name__ == '__main__':
    make_phantom_plot()
    make_grid_plot(sd=2)
    make_grid_plot(sd=2, drop=True)