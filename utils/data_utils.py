import numpy as np
import SimpleITK as sitk
from einops.einops import rearrange, repeat
from scipy.ndimage import gaussian_filter
import torch
def load_data(gaussian_filter_type, sd=2.5, folder=r'data/DigitalPhantomCT'):
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
    perfusion_data = perfusion_data.astype(np.float32)

    if gaussian_filter_type:
        perfusion_data = apply_gaussian_filter(gaussian_filter_type, perfusion_data, sd=sd)
    # exp_data data has shape 15 (curve simulation type *CBV) x 30 (Time) x 224 (7 x delay_step) x 224 (7 x MTT step)

    perfusion_values = np.empty([5, 7, 7, 4])
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
    perfusion_values = repeat(perfusion_values, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    perfusion_data = rearrange(perfusion_data, '(n cbv) t h w -> n cbv h w t', n=3)

    time = np.array([float(x) for x in range(0, 60, 2)])
    # x = x.repeat(self.shape[0], self.shape[1], 1)
    # aif_data = repeat(aif_data, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    # vof_data = repeat(vof_data, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    # time = repeat(time, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time,
                 'curves': perfusion_data[2:,4:,0:1,-64:-32,:],
                 'perfusion_values': perfusion_values[2:, 4:, 0:1, -64:-32,:]}

    data_dict = normalize_data(data_dict)
    data_dict = get_coll_points(data_dict)
    data_dict = get_tensors(data_dict)
    return data_dict


def normalize_data(data_dict):
    # input normalization
    data_dict['std_t'] = data_dict['time'].std()
    data_dict['mean_t'] = data_dict['time'].mean()
    data_dict['time'] = (data_dict['time'] - data_dict['time'].mean()) / data_dict['time'].std()
    # output normalization
    max_ = data_dict['aif'].max()
    data_dict['aif'] /= max_
    data_dict['vof'] /= max_
    data_dict['curves'] /= max_
    return data_dict


def get_coll_points(data_dict):

    data_dict['coll_points'] = np.random.uniform(
        min(data_dict['time']), max(data_dict['time']), len(data_dict['time']) * 5
    ).astype(np.float32)
    data_dict['bound'] = np.array([min(data_dict['time'])])
    return data_dict


def get_tensors(data_dict):
    for key in data_dict.keys():
        data_dict[key] = torch.as_tensor(data_dict[key], dtype=torch.float32)
    return data_dict


def apply_gaussian_filter(type, array, sd):
    assert len(array.shape) == 4
    if type == 'spatio-temporal':
        return gaussian_filter(array, sigma=(0, sd, sd, sd), mode='nearest')
    elif type == 'spatial':
        return gaussian_filter(array, sigma=(0, 0, sd, sd), mode='nearest')
    else:
        raise NotImplementedError('Gaussian filter variant not implemented.')

