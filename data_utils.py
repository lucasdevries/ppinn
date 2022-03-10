import numpy as np
import SimpleITK as sitk
from einops.einops import rearrange, repeat
from scipy.ndimage import gaussian_filter

def load_data(folder=r'data/DigitalPhantomCT'):
    print("Reading Dicom directory:", folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_data = sitk.GetArrayFromImage(image)
    # values: AIF/VOF, Exp R(t) for CBV 1-5, Lin R(t) for CBV 1-5, Box R(t) for CBV 1-5,
    image_data = rearrange(image_data, '(t values) h w -> values t h w', t=30)

    aif_location = (123,251,8) # start, start, size
    aif_data = image_data[0,:,
               aif_location[0]:aif_location[0]+aif_location[2],
               aif_location[1]:aif_location[1]+aif_location[2]]
    aif_data = np.mean(aif_data, axis=(1,2))

    vof_location = (410,247,16) # start, start, size
    vof_data = image_data[0,:,
               vof_location[0]:vof_location[0]+vof_location[2],
               vof_location[1]:vof_location[1]+vof_location[2]]
    vof_data = np.mean(vof_data, axis=(1,2))

    simulated_data_size = 32 * 7
    scan_center = image_data.shape[-1]//2
    simulated_data_start = scan_center - simulated_data_size//2
    simulated_data_end = scan_center + simulated_data_size//2

    perfusion_data = image_data[1:,:,
                     simulated_data_start:simulated_data_end,
                     simulated_data_start:simulated_data_end]
    # exp_data data has shape 5 (CBV) x 30 (Time) x 224 (7 x delay_step) x 224 (7 x MTT step)
    [exp_data, lin_data, box_data] = np.array_split(perfusion_data, 3, axis=0)


    perfusion_values = np.empty([exp_data.shape[0], 7, 7, 4])
    cbv = [1, 2, 3, 4, 5] # in ml / 100g
    mtt_s = [24.0, 12.0, 8.0, 6.0, 4.8, 4.0, 3.42857143] # in seconds
    mtt_m = [t/60 for t in mtt_s] # in minutes
    delay = [0., 0.5, 1., 1.5, 2., 2.5, 3.] # in seconds

    for ix, i in enumerate(cbv):
        for jx, j in enumerate(delay):
            for kx, k in enumerate(mtt_m):
                # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
                values = np.array([i, j, k, i/k])
                perfusion_values[ix, jx, kx] = values
    perfusion_values = repeat(perfusion_values, 'cbv h w values -> cbv (h r1) (w r2) values', r1=32, r2=32)
    time = np.array([float(x) for x in range(0, 60, 2)])
    return {'aif': aif_data,
            'vof': vof_data,
            'time': time,
            'exp_curves': exp_data,
            'lin_curves': lin_data,
            'box_curves': box_data,
            'perfusion_values': perfusion_values}

def gaussian_filter_spatial(array, sd):
    assert len(array.shape) == 3
    return gaussian_filter(array, sigma=(0,sd,sd))

def gaussian_filter_spatial_per_frame(array, sd):
    assert len(array.shape) == 3
    result = np.empty_like(array)
    for frame in range(array.shape[0]):
        result[frame] = gaussian_filter(array[frame], sigma=sd)
    return result