import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np
from utils.val_utils import drop_edges, drop_unphysical
from utils.val_utils import load_sygnovia_results, load_nlr_results, load_phantom_gt, load_sygnovia_results_amc
from utils.plot_utils import bland_altman_plot

font = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 12,
        }
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["figure.dpi"] = 150

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
               'ppinn': get_data(load_phantom_gt, sd=None)}
    if drop:
        results = drop_edges(results)
        results = drop_unphysical(results)

    min_vals = {'cbf':0, 'mtt':0, 'cbv':0, 'delay':0, 'tmax':0}
    max_vals = {'cbf':100, 'mtt':28, 'cbv':7, 'delay':3.5, 'tmax':18}
    title = {'gt': 'GT', 'sygnovia':'Sygno.via', 'nlr':'NLR', 'ppinn':'PPINN'}
    param_title = {'cbf':'CBF', 'mtt':'MTT', 'cbv':'CBV', 'delay':'Delay', 'tmax':'Tmax'}

    x1 = [112 + 224 * x for x in [0, 1, 2, 3, 4]]
    names = ['1 ml ', '2 ml ', '3 ml ', '4 ml ', '5 ml ']
    fig = plt.figure(figsize=(13, 10))
    outer = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)

    for i, param in zip(range(5), ['cbf', 'mtt', 'cbv', 'delay', 'tmax']):
        inner = gridspec.GridSpecFromSubplotSpec(1, 4,
                                                 subplot_spec=outer[i],
                                                 wspace=0.1,
                                                 hspace=0.1)
        ax = plt.Subplot(fig, outer[i])
        ax.set_title(f'{param_title[param]}', size=20, pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
        fig.add_subplot(ax)
        for j,key in zip(range(4), ['gt', 'sygnovia', 'nlr', 'ppinn']):
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


if __name__ == '__main__':
    make_grid_plot(sd=2)