import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np
import os
from utils.val_utils import drop_edges, drop_unphysical, drop_edges_per_method, drop_unphysical_per_method
from utils.val_utils import load_sygnovia_results, load_nlr_results, load_phantom_gt, load_sygnovia_results_amc
from utils.plot_utils import bland_altman_plot
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SimpleITK as sitk
from einops.einops import rearrange, repeat
from tqdm import tqdm
import pandas as pd

def check(case, slice):
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    cases = os.listdir(base)[:-1]
    min_vals = {'cbf': 0, 'mtt': 0, 'cbv': 0, 'delay': 0, 'tmax': 0}
    max_vals = {'cbf': 100, 'mtt': 28, 'cbv': 6, 'delay': 3.5, 'tmax': 20}
    param_title = {'cbf': 'CBF', 'mtt': 'MTT', 'cbv': 'CBV', 'delay': 'Delay', 'tmax': 'Tmax'}
    results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
               'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
               'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
               'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
               'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
               'vesselmask': sitk.ReadImage(os.path.join(base, case, "vesselmask.nii")),
               'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
               'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz")),
               'contra': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                     "DWI_seg_registered_contralateral.nii.gz")),
               'contra': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                     "DWI_seg_registered_contralateral.nii.gz"))
               }

    spacing = results['dwi_seg'].GetSpacing()
    results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}
    plt.imshow(results['mtt'][slice], vmin=0, vmax=7, cmap='jet')
    plt.show()
    curves = []
    for i in range(30):
        i = str(i).zfill(2)
        # sitk_img = sitk.ReadImage(r"D:\PPINN_patient_data\AMCCTP\CTP_nii_registered\C102\00.nii.gz")
        sitk_img = sitk.ReadImage(os.path.join(rf"D:\PPINN_patient_data\AMCCTP\\CTP_nii_registered", case,
                                                     f"{i}.nii.gz"))
        arr = sitk.GetArrayFromImage(sitk_img)
        curves.append(np.mean(arr[slice][160:180,160:180]))
    plt.plot(curves)
    plt.show()



if __name__ == '__main__':
    check('C102', 14)
    check('C102', 15)
    check('C102', 16)
    check('C102', 17)