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


def make_core(parameter):
    base = r"C:\Users\lucasdevries\surfdrive\Projects\ppinn\data\results_sppinn_amcctp"
    cases = os.listdir(base)
    dices = []
    for case in cases:
        print(case)

        results = {'cbv': sitk.ReadImage(os.path.join(base, case, "cbv.nii")),
                   'cbf': sitk.ReadImage(os.path.join(base, case, "cbf.nii")),
                   'mtt': sitk.ReadImage(os.path.join(base, case, "mtt.nii")),
                   'tmax': sitk.ReadImage(os.path.join(base, case, "tmax.nii")),
                   'delay': sitk.ReadImage(os.path.join(base, case, "delay.nii")),
                   'dwi_seg': sitk.ReadImage(os.path.join(base, case, "dwi_seg.nii")),
                   'brainmask': sitk.ReadImage(os.path.join(base, case, "brainmask.nii")),
                   'vesselmask': sitk.ReadImage(os.path.join(base, case, "vesselmask.nii")),

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

        results['RH'] = results['RH'] * results['vesselmask']
        results['LH'] = results['LH'] * results['vesselmask']
        # use only central 50% of data
        start = results[parameter].shape[0]//4
        end = 3*results[parameter].shape[0]//4

        if occl_site[case] == 'L':
            healthy = results['RH'][start:end] * results[parameter][start:end]
            mean_healthy = np.mean(healthy[healthy > 0])
            print(mean_healthy)
        elif occl_site[case] == 'R':
            healthy = results['LH'][start:end] * results[parameter][start:end]
            mean_healthy = np.mean(healthy[healthy > 0])
            print(mean_healthy)
        else:
            raise NotImplementedError('Not possible..')

        results['relative'] = results[parameter] / mean_healthy

        results['pred_core'] = np.zeros_like(results['relative'])
        results['hypoperfused'] = np.zeros_like(results['relative'])
        results['corrected_core'] = np.zeros_like(results['relative'])

        if occl_site[case] == 'L':
            results['hypoperfused'][(results['tmax']>6) & (results['LH'] > 0)] = 1
            results['pred_core'][(results['LH'] > 0) & (results['relative']<0.3)] = 1
        elif occl_site[case] == 'R':
            results['hypoperfused'][(results['tmax']>6) & (results['RH'] > 0)] = 1
            results['pred_core'][(results['RH'] > 0) & (results['relative']<0.3)] = 1
        else:
            raise NotImplementedError('Not possible..')

        results['corrected_core'][(results['pred_core'] == 1) & (results['hypoperfused'] == 1)] = 1

        cbv = sitk.ReadImage(os.path.join(base, case, "cbv.nii"))
        pred_core_sitk = np2itk(results['pred_core'], cbv)
        corrected_core_sitk = np2itk(results['corrected_core'], cbv)
        hypoperfused_sitk = np2itk(results['hypoperfused'], cbv)
        relative_sitk = np2itk(results['relative'], cbv)

        sitk.WriteImage(relative_sitk, os.path.join(base, case, f"r{parameter}.nii.gz"))
        sitk.WriteImage(pred_core_sitk, os.path.join(base, case, f"pred_core_{parameter}.nii.gz"))
        sitk.WriteImage(corrected_core_sitk, os.path.join(base, case, f"corrected_core_{parameter}.nii.gz"))
        sitk.WriteImage(hypoperfused_sitk, os.path.join(base, case, f"hypoperfused.nii.gz"))

        seg = results['pred_core']
        gt = results['dwi_seg'] * results['vesselmask']
        dices.append(np.sum(seg[gt == 1]) * 2.0 / (np.sum(seg) + np.sum(gt)))
    print(dices)
        # if case == 'C106':
        #     for i in range(20):
        #         plt.imshow(results['pred_core'][i])
        #         plt.title(case)
        #         plt.show()







if __name__ == '__main__':
    make_core('cbf')
    make_core('cbv')

