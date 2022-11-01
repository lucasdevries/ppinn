import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y,s=0.1, c=z, cmap='jet', alpha=0.5, **kwargs )
    if ax is None :
        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel('Density')

    return ax

def bland_altman_plot(data1, data2, ax, line=True, xlim=150, ylim=200, label=None, *args, **kwargs):
    max_ = np.max(np.array([data1, data2]))

    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    true_vals = np.unique(data2)
    medians = []
    means = []
    lower_errors = []
    upper_errors = []
    # ensity_scatter(data2, diff, ax, bins=[100, 100])
    for val in true_vals:
        diffs = diff[data2 == val]
        truths = data2[data2 == val]
        estimates_stats = np.percentile(diffs, [2.5, 50, 97.5])
        lower_errors.append(np.abs(estimates_stats[0]))
        medians.append(estimates_stats[1])
        upper_errors.append(np.abs(estimates_stats[2]))
        means.append(np.mean(diffs))
        # density_scatter(truths, diffs, ax, bins=[100, 100])
    asymmetric_error = np.array(list(zip(lower_errors, upper_errors))).T
    density_scatter(data2, diff, ax, bins=[100, 100])
    # ax.errorbar(true_vals, medians, c='k',alpha=0.2, s=0.1, yerr=asymmetric_error, fmt='.', ecolor='k',
    #             barsabove=True, errorevery=1, elinewidth=None,
    #             capsize=2, capthick=2
    #             )

    # ax.scatter(data2, diff, *args, **kwargs)
    #

    # if label:
    #     ax.scatter(data2, diff, label=label, *args, **kwargs)
    # else:
    #     # ax.scatter(data2, diff, *args, **kwargs)
    #     density_scatter(data2, diff, ax, bins=[100, 100])
    #
    # if line:
    #     ax.axhline(md,           color='k', linestyle='--')
    #     ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    #     ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    #     ax.annotate(f"{md:.2f}", xy=(xlim, md), weight="bold")
    #     ax.annotate(f"{md + 1.96*sd:.2f}", xy=(xlim, md + 1.96*sd))
    #     ax.annotate(f"{md - 1.96*sd:.2f}", xy=(xlim, md - 1.96*sd))
    #
    ax.set_xlim(0, xlim)
    ax.set_ylim(-ylim, ylim)

    # def bland_altman_plot(data1, data2, ax, line=True, xlim=150, ylim=200, label=None, *args, **kwargs):
        # max_ = np.max(np.array([data1, data2]))
        #
        # mean = np.mean([data1, data2], axis=0)
        # diff = data1 - data2
        # md = np.mean(diff)
        # sd = np.std(diff, axis=0)
        #
        # if label:
        #     ax.scatter(mean, diff, label=label, *args, **kwargs)
        # else:
        #     ax.scatter(mean, diff, *args, **kwargs)
        #
        # if line:
        #     ax.axhline(md, color='k', linestyle='--')
        #     ax.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        #     ax.axhline(md - 1.96 * sd, color='gray', linestyle='--')
        #     ax.annotate(f"{md:.2f}", xy=(xlim, md), weight="bold")
        #     ax.annotate(f"{md + 1.96 * sd:.2f}", xy=(xlim, md + 1.96 * sd))
        #     ax.annotate(f"{md - 1.96 * sd:.2f}", xy=(xlim, md - 1.96 * sd))
        #
        # ax.set_xlim(-xlim, xlim)
        # ax.set_ylim(-ylim, ylim)