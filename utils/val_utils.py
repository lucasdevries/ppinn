import SimpleITK as sitk
import numpy as np
def load_nlr_results():
    result = sitk.ReadImage(r'L:\basic\divi\CMAJOIE\CLEOPATRA\Substudies\Lucas\KudoPhantom\unfiltered_rescaled_aif.nii')
    result = sitk.GetArrayFromImage(result)
    simulated_data_size = 32 * 7
    scan_center = 512//2
    simulated_data_start = scan_center - simulated_data_size//2
    simulated_data_end = scan_center + simulated_data_size//2

    result = result[...,
             simulated_data_start:simulated_data_end,
             simulated_data_start:simulated_data_end]

    density = 1.05
    constant = (100 / density) * 0.55 / 0.75
    cbv = result[0]
    cbv = cbv * constant
    mtt = result[1]
    cbf = cbv / (mtt / 60)
    delay = result[2]

    return {'cbf': np.transpose(cbf),
            'mtt': np.transpose(mtt),
            'cbv': np.transpose(cbv),
            'delay': np.transpose(delay)}