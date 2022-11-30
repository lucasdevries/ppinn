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
from skimage.filters import gaussian

def get_dice():
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    cases = os.listdir(base)[:-1]
    dices = []
    for case in cases:
        results = {
                   'dwi_core': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                   'sv_core': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\sygnovia", case,
                                                     "nifti\core.nii.gz"))
                   }
        results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}

        seg = results['sv_core']
        gt = results['dwi_core']
        dice = np.sum(seg[gt == 1]) * 2.0 / (np.sum(seg) + np.sum(gt))
        dices.append(dice)
    print(np.mean(dices))
    ppinn = [0.4036775494568242, 0.0, 0.0003404110463384537, 0.5043673378152994, 0.4283956221866095, 0.0681640418560406, 0.065319763025976, 0.24163950402354042, 0.001311921779758037, 0.021643227960710717, 0.39152581087254457, 0.18600388182014232, 0.004432768028412122, 0.29386449549287585, 0.5861949135090609]
    for c, x, y in zip(cases, dices, ppinn):
        print(c, x,y)
    print(np.mean(ppinn))
    plt.boxplot([dices, ppinn])
    plt.show()
if __name__ == '__main__':
    get_dice()
