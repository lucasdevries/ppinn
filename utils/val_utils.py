import SimpleITK as sitk
import numpy as np
def load_nlr_results():
    result = sitk.ReadImage(r'L:\basic\divi\CMAJOIE\CLEOPATRA\Substudies\Lucas\KudoPhantom\unfiltered_rescaled_aif_lucas.nii')
    result = sitk.GetArrayFromImage(result)

    density = 1.05
    constant = (100 / density) * 0.55 / 0.75
    cbv = result[0]
    cbv = cbv * constant
    mtt = result[1]
    cbf = cbv / (mtt / 60)
    delay = result[2]

    return {'cbf': cbf,
            'mtt': mtt,
            'cbv': cbv,
            'delay': delay}