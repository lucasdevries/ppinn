import matplotlib.pyplot as plt
import numpy as np
def bland_altman_plot(data1, data2, ax, line=True, xlim=150, ylim=200, label=None, *args, **kwargs):
    max_ = np.max(np.array([data1, data2]))

    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    if label:
        ax.scatter(mean, diff, label=label, *args, **kwargs)
    else:
        ax.scatter(mean, diff, *args, **kwargs)

    if line:
        ax.axhline(md,           color='k', linestyle='--')
        ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
        ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
        ax.annotate(f"{md:.2f}", xy=(xlim, md), weight="bold")
        ax.annotate(f"{md + 1.96*sd:.2f}", xy=(xlim, md + 1.96*sd))
        ax.annotate(f"{md - 1.96*sd:.2f}", xy=(xlim, md - 1.96*sd))

    ax.set_xlim(0, xlim)
    ax.set_ylim(-ylim, ylim)