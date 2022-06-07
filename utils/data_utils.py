import numpy as np
import SimpleITK as sitk
from einops.einops import rearrange, repeat
from scipy.ndimage import gaussian_filter, convolve
from skimage.restoration import denoise_bilateral
from skimage.filters import gaussian
import torch
import os, glob
import matplotlib.pyplot as plt
import scipy.io
def load_data(gaussian_filter_type, sd=2.5,
              folder=r'data/DigitalPhantomCT',
              cbv_slice=4, simulation_method=2,
              method='ppinn',
              temporal_smoothing=False,
              save_nifti=True,
              baseline_zero=False):
    print("Reading Dicom directory:", folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_data = sitk.GetArrayFromImage(image)
    # values: AIF/VOF, Exp R(t) for CBV 1-5, Lin R(t) for CBV 1-5, Box R(t) for CBV 1-5,
    image_data = rearrange(image_data, '(t values) h w -> values t h w', t=30)
    image_data = image_data.astype(np.float32)
    # if gaussian_filter_type and (method == 'nlr'):
    #     print('filter entire scan')
    #     image_data = apply_gaussian_filter(gaussian_filter_type, image_data.copy(), sd=sd)

    vof_location = (410,247,16) # start, start, size
    vof_data = image_data[0,
               :,
               vof_location[0]:vof_location[0]+vof_location[2],
               vof_location[1]:vof_location[1]+vof_location[2]]
    vof_data = np.mean(vof_data, axis=(1,2))

    aif_location = (123,251,8) # start, start, size
    aif_data = image_data[0,
               :,
               aif_location[0]:aif_location[0]+aif_location[2],
               aif_location[1]:aif_location[1]+aif_location[2]]
    aif_data = np.mean(aif_data, axis=(1,2))
    if method == 'ppinn' or method == 'nlr':
        # Correct aif for partial volume effect
        vof_baseline = np.mean(vof_data[:4])
        aif_baseline = np.mean(aif_data[:4])
        aif_wo_baseline = aif_data - aif_baseline
        vof_wo_baseline = vof_data - vof_baseline
        cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
        cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
        ratio = cumsum_vof / cumsum_aif
        aif_data = aif_wo_baseline * ratio + aif_baseline

    if baseline_zero:
        aif_baseline = np.mean(aif_data[:4])
        aif_data = aif_data - aif_baseline
        tac_baseline = np.expand_dims(np.mean(image_data[:,:4,...], axis=1), axis=1)
        image_data = image_data - tac_baseline

    simulated_data_size = 32 * 7
    scan_center = image_data.shape[-1]//2
    simulated_data_start = scan_center - simulated_data_size//2
    simulated_data_end = scan_center + simulated_data_size//2
    if gaussian_filter_type:
        perfusion_data = image_data[1:,:,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
        perfusion_data = apply_gaussian_filter(gaussian_filter_type, perfusion_data.copy(), sd=sd)

    else:
        perfusion_data = image_data[1:,:,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
    perfusion_data = perfusion_data.astype(np.float32)


    if save_nifti:
        scipy.io.savemat(os.path.join('data', 'aif_data.mat'), {'aif':aif_data})
        perfusion_data_nii = rearrange(perfusion_data, 'c t h w -> h w c t')
        scipy.io.savemat(os.path.join('data', 'image_data_sd_{}.mat'.format(sd)), {'image_data': perfusion_data_nii})
        perfusion_data_nii = sitk.GetImageFromArray(perfusion_data_nii)
        sitk.WriteImage(perfusion_data_nii, os.path.join('data', 'image_data_sd_{}.nii'.format(sd)))


    if temporal_smoothing:
        k = np.array([0.25, 0.5, 0.25])
        aif_data = convolve(aif_data, k, mode='nearest')
        k = k.reshape(1,3,1,1)
        perfusion_data = convolve(perfusion_data, k, mode='nearest')

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

    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time,
                 'curves': perfusion_data[simulation_method:simulation_method+1, cbv_slice:cbv_slice+1, :, :, :],
                 'perfusion_values': perfusion_values[simulation_method:simulation_method+1, cbv_slice:cbv_slice+1, :, :, :]}

    if method == 'ppinn':
        data_dict = normalize_data(data_dict)
        data_dict = get_coll_points(data_dict)
        data_dict = get_tensors(data_dict)
    elif method == 'nlr':
        print('Data mode: nlr...')
    else:
        raise NotImplementedError('Data method not implemented.')
    return data_dict

def load_data_ISLES(filter_type, sd=2.5,
                    folder=r'data/ISLES2018',
                    folder_aif=r'data/AIFNET',

                    method='ppinn',
                    temporal_smoothing=False,
                    save_nifti=True,
                    baseline_zero=False,
                    mode='TRAINING',
                    case=3):
    print("Data folder: {}.".format(folder))
    print("Reading {} data, case {}.".format(mode, case), folder)

    image_data_path = os.path.join(folder, mode, f'case_{case}', '*PWI*', '*PWI*.nii')
    image_data_path = glob.glob(image_data_path)[0]
    image = sitk.ReadImage(image_data_path)
    image_data = sitk.GetArrayFromImage(image)
    image_data = rearrange(image_data, 't d h w -> d t h w')
    image_data = image_data.astype(np.float32)

    vof_data_path = os.path.join(folder_aif, mode, 'VOF', f'case_{case}.npy')
    aif_data_path = os.path.join(folder_aif, mode, 'AIF', f'case_{case}.npy')
    vof_data = np.load(vof_data_path)
    aif_data = np.load(aif_data_path)

    if method == 'ppinn' or method == 'nlr':
        # Correct aif for partial volume effect
        vof_baseline = np.mean(vof_data[:4])
        aif_baseline = np.mean(aif_data[:4])
        aif_wo_baseline = aif_data - aif_baseline
        vof_wo_baseline = vof_data - vof_baseline
        cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
        cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
        ratio = cumsum_vof / cumsum_aif
        aif_data = aif_wo_baseline * ratio + aif_baseline

    if baseline_zero:
        aif_baseline = np.mean(aif_data[:4])
        aif_data = aif_data - aif_baseline
        tac_baseline = np.expand_dims(np.mean(image_data[:,:4,...], axis=1), axis=1)
        image_data = image_data - tac_baseline

    perfusion_map_data_paths = [os.path.join(folder, mode, f'case_{case}', '*CBF*', '*CBF*.nii'),
                                os.path.join(folder, mode, f'case_{case}', '*CBV*', '*CBV*.nii'),
                                os.path.join(folder, mode, f'case_{case}', '*MTT*', '*MTT*.nii'),
                                os.path.join(folder, mode, f'case_{case}', '*Tmax*', '*Tmax*.nii'),
                                os.path.join(folder, mode, f'case_{case}', '*OT*', '*OT*.nii')]
    perfusion_map_data_paths = [glob.glob(path)[0] for path in perfusion_map_data_paths]
    perfusion_map_data = sitk.ReadImage(perfusion_map_data_paths[0])
    perfusion_map_data = sitk.GetArrayFromImage(perfusion_map_data)
    brainmask = np.zeros_like(perfusion_map_data, dtype=np.float32)
    brainmask += perfusion_map_data
    for perfusion_map in perfusion_map_data_paths[1:]:
        perfusion_map_data = sitk.ReadImage(perfusion_map)
        perfusion_map_data = sitk.GetArrayFromImage(perfusion_map_data)
        brainmask += perfusion_map_data
    brainmask = (brainmask > 0).astype(float)
    perfusion_values = np.zeros([*brainmask.shape, 5], dtype=np.float32)

    for ix, perfusion_map in enumerate(perfusion_map_data_paths):
        perfusion_map_data = sitk.ReadImage(perfusion_map)
        perfusion_map_data = sitk.GetArrayFromImage(perfusion_map_data)
        perfusion_values[..., ix] = perfusion_map_data

    if filter_type == 'gauss_spatial' or filter_type =='gauss_spatiotemporal' :
        perfusion_data = apply_gaussian_filter_with_mask(filter_type, image_data, brainmask, sd=sd)
    elif filter_type == 'billateral':
        perfusion_data = apply_billateral_filter(image_data, brainmask, sigma_spatial=sd)
    else:
        perfusion_data = image_data
    perfusion_data = perfusion_data.astype(np.float32)

    if temporal_smoothing:
        k = np.array([0.25, 0.5, 0.25])
        aif_data = convolve(aif_data, k, mode='nearest')
        k = k.reshape(1,3,1,1)
        perfusion_data = convolve(perfusion_data, k, mode='nearest')

    time = np.array([float(x) for x in range(0, perfusion_data.shape[1], 1)])
    perfusion_data = rearrange(perfusion_data, 'd t h w -> d h w t')


    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time,
                 'curves': perfusion_data,
                 'perfusion_values': perfusion_values,
                 'brainmask': brainmask}

    if method == 'ppinn':
        data_dict = normalize_data(data_dict)
        data_dict = get_coll_points(data_dict)
        data_dict = get_tensors(data_dict)
    elif method == 'nlr':
        print('Data mode: nlr...')
    else:
        raise NotImplementedError('Data method not implemented.')
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
        min(data_dict['time']), max(data_dict['time']), len(data_dict['time']) * 5 * 10 * 3
    ).astype(np.float32)
    data_dict['bound'] = np.array([min(data_dict['time'])])
    return data_dict


def get_tensors(data_dict):
    for key in data_dict.keys():
        data_dict[key] = torch.as_tensor(data_dict[key], dtype=torch.float32)
    return data_dict


def apply_gaussian_filter(type, array, sd):
    truncate = np.ceil(2 * sd) / sd if sd != 0 else 0
    if len(array.shape) == 4:
        if type == 'gauss_spatiotemporal':
            return gaussian(array, sigma=(0, sd, sd, sd), mode='nearest', truncate=truncate)
        elif type == 'gauss_spatial':
            return gaussian(array, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
        else:
            raise NotImplementedError('Gaussian filter variant not implemented.')

    if len(array.shape) == 3:
        if type == 'gauss_spatiotemporal':
            return gaussian(array, sigma=(sd, sd, sd), mode='nearest', truncate=truncate)
        elif type == 'gauss_spatial':
            return gaussian(array, sigma=(0, sd, sd), mode='nearest', truncate=truncate)


def apply_gaussian_filter_with_mask(type, array, mask, sd):
    truncate = np.ceil(2 * sd) / sd if sd != 0 else 0
    mask = np.expand_dims(mask, 1)
    mask = np.repeat(mask, array.shape[1], axis=1)
    if len(array.shape) == 4:
        if type == 'gauss_spatiotemporal':
            filtered = gaussian(array * mask, sigma=(0, sd, sd, sd), mode='nearest', truncate=truncate)
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered /= gaussian(mask, sigma=(0, sd, sd, sd), mode='nearest', truncate=truncate)
            filtered[np.logical_not(mask)] = 0
            return filtered
        elif type == 'gauss_spatial':
            filtered = gaussian(array * mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            intermed = gaussian(mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            with np.errstate(divide='ignore', invalid='ignore'):
                filtered /= gaussian(mask, sigma=(0, 0, sd, sd), mode='nearest', truncate=truncate)
            filtered[np.logical_not(mask)] = 0
            return filtered
        else:
            raise NotImplementedError('Gaussian filter variant not implemented.')

def apply_billateral_filter(array, mask, sigma_spatial):
    mask = np.expand_dims(mask, 1)
    mask = np.repeat(mask, array.shape[1], axis=1)
    filtered = np.zeros_like(array, dtype=np.float32)
    plt.imshow(array[4,0], vmin=0, vmax=100)
    plt.show()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filtered[i,j] = denoise_bilateral(array[i,j], win_size=None,
                                              sigma_color=20, sigma_spatial=sigma_spatial,
                                              bins=10000)
    filtered *= mask
    plt.imshow(filtered[4,0], vmin=0, vmax=100)
    plt.show()
    return filtered