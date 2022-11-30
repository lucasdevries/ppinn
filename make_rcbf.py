import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np
import os
from utils.data_utils import np2itk
from utils.val_utils import drop_edges, drop_unphysical
from utils.val_utils import load_sygnovia_results, load_nlr_results, load_phantom_gt, load_sygnovia_results_amc
from utils.plot_utils import bland_altman_plot
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SimpleITK as sitk
from einops.einops import rearrange, repeat
from skimage.filters import gaussian

occl_site = {
    'C102': 'L',
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
    'C116': 'R',
}


def make_rcbf():
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_ppinn_amcctp"
    cases = os.listdir(base)[:-1]
    dices = []

    for case in cases[-3:]:
        print(case)
        results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                   'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                   'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                   'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                   'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                   'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                   'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii.gz")),
                   'mip': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MIP", case, "mip.nii.gz")),
                   'contra': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                         "DWI_seg_registered_contralateral.nii.gz")),
                   'LH': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                     "DWI_seg_registered_LH.nii.gz")),
                   'RH': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\MRI_nii_registered", case,
                                                     "DWI_seg_registered_RH.nii.gz")),
                   'baseline': sitk.ReadImage(os.path.join(r"D:\PPINN_patient_data\AMCCTP\CTP_nii_brainmask", case,
                                                     "00_noskull.nii.gz")),
                   }
        results = {key: sitk.GetArrayFromImage(val) for key, val in results.items()}
        sd = 2
        truncate = np.ceil(2 * sd) / sd if sd != 0 else 0
        results['baseline'] = gaussian(results['baseline'], sigma=(0, sd, sd), mode='nearest', truncate=truncate)

        results['tissue_mask'] = np.zeros_like(results['baseline'])
        results['tissue_mask'][(results['baseline'] > 30) & (results['baseline'] < 100) & (results['cbf'] > 0)] = 1

        sd = 5
        truncate = np.ceil(2 * sd) / sd if sd != 0 else 0
        results['cbf'] = gaussian(results['cbf'], sigma=(0, sd, sd), mode='nearest', truncate=truncate)
        results['cbf'] = results['tissue_mask'] * results['cbf']
        # plt.imshow(results['cbf'][22])
        # plt.title(case)
        # plt.show()

        results['RH'] = results['RH'] * results['tissue_mask']
        results['LH'] = results['LH'] * results['tissue_mask']

        if occl_site[case] == 'L':
            healthy = results['RH'] * results['cbf']
            mean_healthy = np.mean(healthy[healthy > 0])
            print(mean_healthy)
        elif occl_site[case] == 'R':
            healthy = results['LH'] * results['cbf']
            mean_healthy = np.mean(healthy[healthy > 0])
            print(mean_healthy)

        else:
            raise NotImplementedError('Not possible..')

        plt.hist(healthy[healthy > 0].flatten(), bins=50)
        plt.show()

        results['rcbf'] = results['cbf'] / mean_healthy

        results['pred_core'] = np.zeros_like(results['rcbf'])

        if occl_site[case] == 'L':
            results['pred_core'][(results['LH'] > 0) & (results['rcbf']<0.3)] = 1
        elif occl_site[case] == 'R':
            results['pred_core'][(results['RH'] > 0) & (results['rcbf']<0.3)] = 1
        else:
            raise NotImplementedError('Not possible..')

        cbv = sitk.ReadImage(os.path.join(base, case, "cbv.nii"))
        pred_sitk = np2itk(results['pred_core'],cbv)
        sitk.WriteImage(pred_sitk, os.path.join(base, case, "pred_ppinn.nii.gz"))
        rcbf_sitk = np2itk(results['rcbf'],cbv)
        sitk.WriteImage(rcbf_sitk, os.path.join(base, case, "rcbf.nii.gz"))

        #
        seg = results['pred_core']
        gt = results['dwi_seg']
        dices.append(np.sum(seg[gt == 1]) * 2.0 / (np.sum(seg) + np.sum(gt)))
    print(dices)
        # if case == 'C106':
        #     for i in range(20):
        #         plt.imshow(results['pred_core'][i])
        #         plt.title(case)
        #         plt.show()







if __name__ == '__main__':
    make_rcbf()
