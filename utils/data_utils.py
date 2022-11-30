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
import wandb
from scipy.integrate import simpson
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
def load_data(gaussian_filter_type, sd=2.5,
              folder=r'data/DigitalPhantomCT',
              cbv_ml=5, simulation_method=2,
              method='ppinn',
              temporal_smoothing=False,
              save_nifti=False,
              baseline_zero=False,
              undersampling=0.0):
    print("Reading Dicom directory:", folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_data = sitk.GetArrayFromImage(image)
    # values: AIF/VOF, Exp R(t) for CBV 1-5, Lin R(t) for CBV 1-5, Box R(t) for CBV 1-5,
    image_data = rearrange(image_data, '(t values) h w -> values t h w', t=30)

    image_data = image_data.astype(np.float32)
    time = np.array([float(x) for x in range(0, 60, 2)])
    if undersampling:
        image_data, time = undersample(image_data, time, undersampling)
    # if gaussian_filter_type and (method == 'nlr'):
    #     print('filter entire scan')
    #     image_data = apply_gaussian_filter(gaussian_filter_type, image_data.copy(), sd=sd)

    vof_location = (410, 247, 16)  # start, start, size
    vof_data = image_data[0,
               :,
               vof_location[0]:vof_location[0] + vof_location[2],
               vof_location[1]:vof_location[1] + vof_location[2]]
    vof_data = np.mean(vof_data, axis=(1, 2))

    aif_location = (123, 251, 8)  # start, start, size
    aif_data = image_data[0,
               :,
               aif_location[0]:aif_location[0] + aif_location[2],
               aif_location[1]:aif_location[1] + aif_location[2]]
    aif_data = np.mean(aif_data, axis=(1, 2))
    if method == 'ppinn' or method == 'nlr':
        # Correct aif for partial volume effect
        vof_baseline = np.mean(vof_data[:4]) if undersampling == 0.0 else np.mean(vof_data[:2])
        aif_baseline = np.mean(aif_data[:4]) if undersampling == 0.0 else np.mean(vof_data[:2])
        if undersampling == 0.25:
            aif_baseline = np.mean(aif_data[:1])
            vof_baseline = np.mean(vof_data[:1])
        aif_wo_baseline = aif_data - aif_baseline
        vof_wo_baseline = vof_data - vof_baseline
        cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
        cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
        ratio = cumsum_vof / cumsum_aif
        aif_data = aif_wo_baseline * ratio + aif_baseline
        print(aif_data)

    if baseline_zero:
        aif_baseline = np.mean(aif_data[:4])
        aif_data = aif_data - aif_baseline
        tac_baseline = np.expand_dims(np.mean(image_data[:, :4, ...], axis=1), axis=1)
        image_data = image_data - tac_baseline

    simulated_data_size = 32 * 7
    scan_center = image_data.shape[-1] // 2
    simulated_data_start = scan_center - simulated_data_size // 2
    simulated_data_end = scan_center + simulated_data_size // 2
    if gaussian_filter_type:
        perfusion_data = image_data[1:, :,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
        perfusion_data = apply_gaussian_filter(gaussian_filter_type, perfusion_data.copy(), sd=sd)

    else:
        perfusion_data = image_data[1:, :,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
    perfusion_data = perfusion_data.astype(np.float32)

    if save_nifti:
        scipy.io.savemat(os.path.join('data', 'aif_data.mat'), {'aif': aif_data})
        perfusion_data_nii = rearrange(perfusion_data, 'c t h w -> h w c t')
        # scipy.io.savemat(os.path.join('data', 'image_data_sd_{}.mat'.format(sd)), {'image_data': perfusion_data_nii})
        perfusion_data_nii = sitk.GetImageFromArray(perfusion_data_nii)
        sitk.WriteImage(perfusion_data_nii, os.path.join('data', 'NLR_image_data_sd_{}.nii'.format(sd)))

    if temporal_smoothing:
        k = np.array([0.25, 0.5, 0.25])
        aif_data = convolve(aif_data, k, mode='nearest')
        k = k.reshape(1, 3, 1, 1)
        perfusion_data = convolve(perfusion_data, k, mode='nearest')

    perfusion_values = np.empty([5, 7, 7, 4])
    cbv = [1, 2, 3, 4, 5]  # in ml / 100g
    mtt_s = [24.0, 12.0, 8.0, 6.0, 4.8, 4.0, 3.42857143]  # in seconds
    mtt_m = [t / 60 for t in mtt_s]  # in minutes
    delay = [0., 0.5, 1., 1.5, 2., 2.5, 3.]  # in seconds

    for ix, i in enumerate(cbv):
        for jx, j in enumerate(delay):
            for kx, k in enumerate(mtt_m):
                # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
                values = np.array([i, j, k, i / k])
                perfusion_values[ix, jx, kx] = values
    perfusion_values = repeat(perfusion_values, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    perfusion_values_dict = {'cbf': perfusion_values[simulation_method, cbv_ml-1, ..., 3],
                             'delay': perfusion_values[simulation_method, cbv_ml-1, ..., 1],
                             'cbv': perfusion_values[simulation_method, cbv_ml-1, ..., 0],
                             'mtt': perfusion_values[simulation_method, cbv_ml-1, ..., 2] * 60}
    perfusion_values_dict['tmax'] = perfusion_values_dict['delay'] + 0.5 * perfusion_values_dict['mtt']
    perfusion_data = rearrange(perfusion_data, '(n cbv) t h w -> n cbv h w t', n=3)

    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time,
                 'curves': perfusion_data[simulation_method:simulation_method + 1, cbv_ml-1:cbv_ml, :, :, :],
                 'perfusion_values': perfusion_values[simulation_method:simulation_method + 1, cbv_ml-1:cbv_ml,
                                     :, :, :]}

    if method == 'ppinn':
        data_dict = normalize_data(data_dict)
        data_dict = get_coll_points(data_dict)
        data_dict = get_tensors(data_dict)
    elif method == 'nlr':
        print('Data mode: nlr...')
    else:
        raise NotImplementedError('Data method not implemented.')

    # data_dict = undersample(data_dict, undersampling)

    data_dict['perfusion_values_dict'] = perfusion_values_dict
    return data_dict


def load_data_ISLES(filter_type, sd=2.5,
                    folder=r'data/ISLES2018',
                    folder_aif=r'data/AIFNET',
                    method='ppinn',
                    temporal_smoothing=False,
                    save_nifti=False,
                    baseline_zero=False,
                    mode='train',
                    case='case_3'):
    print("Data folder: {}.".format(folder))
    mode = 'TRAINING' if mode == 'train' else 'TESTING'

    print("Reading {} data, {}.".format(mode, case), folder)

    image_data_path = os.path.join(folder, mode, f'{case}', '*PWI*', '*PWI*.nii')
    image_data_path = glob.glob(image_data_path)[0]
    image = sitk.ReadImage(image_data_path)
    image_data = sitk.GetArrayFromImage(image)
    image_data = rearrange(image_data, 't d h w -> d t h w')
    image_data = image_data.astype(np.float32)

    vof_data_path = os.path.join(folder_aif, mode, 'VOF', f'{case}.npy')
    aif_data_path = os.path.join(folder_aif, mode, 'AIF', f'{case}.npy')
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
        tac_baseline = np.expand_dims(np.mean(image_data[:, :4, ...], axis=1), axis=1)
        image_data = image_data - tac_baseline

    perfusion_map_data_paths = [os.path.join(folder, mode, case, '*CBF*', '*CBF*.nii'),
                                os.path.join(folder, mode, case, '*CBV*', '*CBV*.nii'),
                                os.path.join(folder, mode, case, '*MTT*', '*MTT*.nii'),
                                os.path.join(folder, mode, case, '*Tmax*', '*Tmax*.nii'),
                                os.path.join(folder, mode, case, '*OT*', '*OT*.nii')]
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
    perfusion_values = np.zeros([*brainmask.shape, 6], dtype=np.float32)

    for ix, perfusion_map in enumerate(perfusion_map_data_paths):
        perfusion_map_data = sitk.ReadImage(perfusion_map)
        perfusion_map_data = sitk.GetArrayFromImage(perfusion_map_data)
        perfusion_values[..., ix] = perfusion_map_data
    perfusion_values[...,5] = perfusion_values[...,3] - 0.5 * perfusion_values[...,2]

    if filter_type == 'gauss_spatial' or filter_type == 'gauss_spatiotemporal':
        perfusion_data = apply_gaussian_filter_with_mask(filter_type, image_data, brainmask, sd=sd)
    elif filter_type == 'billateral':
        perfusion_data = apply_billateral_filter(image_data, brainmask, sigma_spatial=sd)
    else:
        perfusion_data = image_data
    perfusion_data = perfusion_data.astype(np.float32)

    if temporal_smoothing:
        k = np.array([0.25, 0.5, 0.25])
        aif_data = convolve(aif_data, k, mode='nearest')
        k = k.reshape(1, 3, 1, 1)
        perfusion_data = convolve(perfusion_data, k, mode='nearest')

    time = np.array([float(x) for x in range(0, perfusion_data.shape[1], 1)])
    time_inference_highres = np.array([float(x) for x in np.arange(0, 60, 0.1)])

    perfusion_data = rearrange(perfusion_data, 'd t h w -> d h w t')

    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time,
                 'time_inference_highres': time_inference_highres,
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


def load_data_AMC(gaussian_filter_type, sd=2,
                    folder=r'D:/PPINN_patient_data/AMCCTP',
                    case='C102'):
    aif_location = os.path.join(folder, rf'AIF_annotations/{case}/aif.nii.gz')
    vof_location = os.path.join(folder, rf'VOF_annotations/{case}/vof.nii.gz')
    time_matrix = os.path.join(folder, rf'CTP_time_matrix/{case}/matrix.npy')
    ctp_folder = os.path.join(folder, rf'CTP_nii_registered/{case}/*.nii.gz')
    brainmask = os.path.join(folder, rf'CTP_nii_brainmask/{case}/brainmask.nii.gz')
    dwi_segmentation = os.path.join(folder,  rf'MRI_nii_registered/{case}/DWI_seg_registered_corrected.nii.gz')
    # load image data
    image_data_dict = read_nii_folder(ctp_folder)
    dwi_segmentation = sitk.ReadImage(dwi_segmentation)
    # load time matrix
    time_data = np.load(time_matrix)
    # load aif and vof locations
    aif_location = sitk.GetArrayFromImage(sitk.ReadImage(aif_location))
    vof_location = sitk.GetArrayFromImage(sitk.ReadImage(vof_location))
    time_aif_location = list(set(np.where(aif_location == 1)[0]))[0]
    time_vof_location = list(set(np.where(vof_location == 1)[0]))[0]
    # load brainmask
    brainmask_data = sitk.GetArrayFromImage(sitk.ReadImage(brainmask))
    # get aif and vof data
    aif_data = np.sum(np.multiply(aif_location, image_data_dict['array']), axis=(1,2,3)) / np.sum(aif_location)
    vof_data = np.sum(np.multiply(vof_location, image_data_dict['array']), axis=(1,2,3)) / np.sum(vof_location)
    aif_time_data = time_data[time_aif_location]
    vof_time_data = time_data[time_vof_location]
    # plt.plot(aif_time_data, aif_data, label='aif')
    # plt.plot(vof_time_data,vof_data, label='vof')
    # plt.title(str(case)+'')
    # plt.legend()
    # plt.savefig(str(case)+'.png')
    # plt.show()
    # scale aif
    vof_baseline = np.mean(vof_data[:4])
    aif_baseline = np.mean(aif_data[:4])
    aif_wo_baseline = aif_data - aif_baseline
    aif_part_nonzero_baseline = aif_wo_baseline.clip(max=0)
    aif_wo_baseline = aif_wo_baseline.clip(min=0)
    vof_wo_baseline = vof_data - vof_baseline
    vof_wo_baseline = vof_wo_baseline.clip(min=0)
    # plt.plot(aif_time_data, aif_wo_baseline, label='aif')
    # plt.plot(vof_time_data, vof_wo_baseline, label='vof')
    # plt.title(str(case)+' nobaseline')
    # plt.legend()
    # plt.show()
    # now we use simpsons approximation because of irregular timing
    cumsum_aif = simpson(aif_wo_baseline, aif_time_data)
    cumsum_vof = simpson(vof_wo_baseline, vof_time_data)
    # print(case, cumsum_aif, cumsum_vof)
    # print(aif_wo_baseline)
    # print(vof_wo_baseline)
    # cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
    # cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
    ratio = cumsum_vof / cumsum_aif
    # cumsum_aif_scaled = simpson(aif_wo_baseline* ratio, aif_time_data)
    # plt.plot(aif_time_data, aif_wo_baseline* ratio, label='aif')
    # plt.plot(vof_time_data, vof_wo_baseline, label='vof')
    # plt.title(str(case)+' nobaseline_scaled')
    # plt.legend()
    # plt.show()
    # print(case, cumsum_aif_scaled, cumsum_vof)

    aif_data = aif_wo_baseline * ratio + aif_baseline
    aif_data += aif_part_nonzero_baseline * ratio

    # plt.plot(aif_time_data, aif_data, label='aif')
    # plt.plot(vof_time_data,vof_data, label='vof')
    # plt.title(str(case)+' norm')
    # plt.legend()
    # plt.savefig(str(case)+'_norm_new.png')
    # plt.show()
    image_data_dict['array'] = np.multiply(image_data_dict['array'], brainmask_data)
    image_data_dict['mip'] = np.max(image_data_dict['array'], axis=0)

    vesselmask = np.zeros_like(image_data_dict['mip'])
    vesselmask[image_data_dict['mip']>150] = 1

    tissuemask = np.zeros_like(image_data_dict['mip'])
    image_data_avg = np.mean(image_data_dict['array'][:4,...], axis=0)
    tissuemask[(image_data_avg > 30) & (image_data_avg < 100)] = 1

    complete_mask = np.zeros_like(image_data_dict['mip'])
    valid_voxels = np.where((brainmask_data == 1) & (tissuemask == 1) & (vesselmask == 0))

    complete_mask[valid_voxels] = 1
    # If smoothing, apply here
    if gaussian_filter_type:
        image_data_dict['array'] = apply_gaussian_filter_with_mask(gaussian_filter_type,
                                                                   image_data_dict['array'].copy(),
                                                                   complete_mask,
                                                                   sd=sd)
    image_data_dict['array'] = image_data_dict['array'].astype(np.float32)
    image_data_dict['array'] = rearrange(image_data_dict['array'], 't d h w -> d h w t')


    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time_data,
                 'curves': image_data_dict['array'],
                 'brainmask': brainmask_data,
                 'mip': image_data_dict['mip'],
                 'mask': complete_mask,
                 }

    data_dict = normalize_data(data_dict)
    data_dict = get_coll_points(data_dict)
    data_dict = get_tensors(data_dict)
    data_dict['aif_time'] = data_dict['time'][time_aif_location]
    data_dict['dwi_segmentation'] = dwi_segmentation
    return data_dict

def load_data_spatiotemporal(gaussian_filter_type, sd=2.5,
              folder=r'data/DigitalPhantomCT',
              cbv_ml=5, simulation_method=2,
              method='ppinn',
              temporal_smoothing=False,
              save_nifti=False,
              baseline_zero=False,
              undersampling=0.0):
    print("Reading Dicom directory:", folder)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_data = sitk.GetArrayFromImage(image)
    # values: AIF/VOF, Exp R(t) for CBV 1-5, Lin R(t) for CBV 1-5, Box R(t) for CBV 1-5,
    image_data = rearrange(image_data, '(t values) h w -> values t h w', t=30)

    image_data = image_data.astype(np.float32)
    time = np.array([float(x) for x in range(0, 60, 2)])
    if undersampling:
        image_data, time = undersample(image_data, time, undersampling)
    vof_location = (410, 247, 16)  # start, start, size
    vof_data = image_data[0,
               :,
               vof_location[0]:vof_location[0] + vof_location[2],
               vof_location[1]:vof_location[1] + vof_location[2]]
    vof_data = np.mean(vof_data, axis=(1, 2))

    aif_location = (123, 251, 8)  # start, start, size
    aif_data = image_data[0,
               :,
               aif_location[0]:aif_location[0] + aif_location[2],
               aif_location[1]:aif_location[1] + aif_location[2]]
    aif_data = np.mean(aif_data, axis=(1, 2))
    if method == 'ppinn' or method == 'nlr':
        # Correct aif for partial volume effect
        vof_baseline = np.mean(vof_data[:4]) if undersampling == 0.0 else np.mean(vof_data[:2])
        aif_baseline = np.mean(aif_data[:4]) if undersampling == 0.0 else np.mean(vof_data[:2])
        if undersampling == 0.25:
            aif_baseline = np.mean(aif_data[:1])
            vof_baseline = np.mean(vof_data[:1])
        aif_wo_baseline = aif_data - aif_baseline
        vof_wo_baseline = vof_data - vof_baseline
        cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
        cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
        ratio = cumsum_vof / cumsum_aif
        aif_data = aif_wo_baseline * ratio + aif_baseline
        # print(aif_data)

    if baseline_zero:
        aif_baseline = np.mean(aif_data[:4])
        aif_data = aif_data - aif_baseline
        tac_baseline = np.expand_dims(np.mean(image_data[:, :4, ...], axis=1), axis=1)
        image_data = image_data - tac_baseline

    simulated_data_size = 32 * 7
    scan_center = image_data.shape[-1] // 2
    simulated_data_start = scan_center - simulated_data_size // 2
    simulated_data_end = scan_center + simulated_data_size // 2
    if gaussian_filter_type:
        perfusion_data = image_data[1:, :,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
        perfusion_data = apply_gaussian_filter(gaussian_filter_type, perfusion_data.copy(), sd=sd)

    else:
        perfusion_data = image_data[1:, :,
                         simulated_data_start:simulated_data_end,
                         simulated_data_start:simulated_data_end]
    perfusion_data = perfusion_data.astype(np.float32)

    if save_nifti:
        scipy.io.savemat(os.path.join('data', 'aif_data.mat'), {'aif': aif_data})
        perfusion_data_nii = rearrange(perfusion_data, 'c t h w -> h w c t')
        # scipy.io.savemat(os.path.join('data', 'image_data_sd_{}.mat'.format(sd)), {'image_data': perfusion_data_nii})
        perfusion_data_nii = sitk.GetImageFromArray(perfusion_data_nii)
        sitk.WriteImage(perfusion_data_nii, os.path.join('data', 'NLR_image_data_sd_{}.nii'.format(sd)))

    if temporal_smoothing:
        k = np.array([0.25, 0.5, 0.25])
        aif_data = convolve(aif_data, k, mode='nearest')
        k = k.reshape(1, 3, 1, 1)
        perfusion_data = convolve(perfusion_data, k, mode='nearest')

    perfusion_values = np.empty([5, 7, 7, 4])
    cbv = [1, 2, 3, 4, 5]  # in ml / 100g
    mtt_s = [24.0, 12.0, 8.0, 6.0, 4.8, 4.0, 3.42857143]  # in seconds
    mtt_m = [t / 60 for t in mtt_s]  # in minutes
    delay = [0., 0.5, 1., 1.5, 2., 2.5, 3.]  # in seconds

    for ix, i in enumerate(cbv):
        for jx, j in enumerate(delay):
            for kx, k in enumerate(mtt_m):
                # 0:'cbv', 1:'delay', 2:'mtt_m', 3:'cbf'
                values = np.array([i, j, k, i / k])
                perfusion_values[ix, jx, kx] = values
    perfusion_values = repeat(perfusion_values, 'cbv h w values -> n cbv (h r1) (w r2) values', n=3, r1=32, r2=32)
    perfusion_values_dict = {'cbf': perfusion_values[simulation_method, cbv_ml-1, ..., 3],
                             'delay': perfusion_values[simulation_method, cbv_ml-1, ..., 1],
                             'cbv': perfusion_values[simulation_method, cbv_ml-1, ..., 0],
                             'mtt': perfusion_values[simulation_method, cbv_ml-1, ..., 2] * 60}
    perfusion_values_dict['tmax'] = perfusion_values_dict['delay'] + 0.5 * perfusion_values_dict['mtt']
    perfusion_data = rearrange(perfusion_data, '(n cbv) t h w -> n cbv h w t', n=3)

    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time,
                 'curves': perfusion_data[simulation_method:simulation_method + 1, cbv_ml-1:cbv_ml, :, :, :],
                 'perfusion_values': perfusion_values[simulation_method:simulation_method + 1, cbv_ml-1:cbv_ml,
                                     :, :, :]}

    if method == 'ppinn':
        data_dict = normalize_data(data_dict)
        data_dict = get_coll_points(data_dict)
        data_dict = get_tensors(data_dict)
    elif method == 'nlr':
        print('Data mode: nlr...')
    else:
        raise NotImplementedError('Data method not implemented.')


    data_dict['time'] = np.tile(
        data_dict['time'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, 224, 224,1),
    ).astype(np.float32)
    data_dict['time_inference_highres'] = np.tile(
        data_dict['time_inference_highres'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, 224, 224,1),
    ).astype(np.float32)
    # create meshes
    data_dict = create_mesh(data_dict)
    data_dict = get_tensors(data_dict)
    data_dict['perfusion_values_dict'] = perfusion_values_dict

    return data_dict

def load_data_AMC_spatiotemporal(gaussian_filter_type, sd=2,
                    folder=r'D:/PPINN_patient_data/AMCCTP',
                    case='C102'):
    aif_location = os.path.join(folder, rf'AIF_annotations/{case}/aif.nii.gz')
    vof_location = os.path.join(folder, rf'VOF_annotations/{case}/vof.nii.gz')
    time_matrix = os.path.join(folder, rf'CTP_time_matrix/{case}/matrix.npy')
    ctp_folder = os.path.join(folder, rf'CTP_nii_registered/{case}/*.nii.gz')
    brainmask = os.path.join(folder, rf'CTP_nii_brainmask/{case}/brainmask.nii.gz')
    dwi_segmentation = os.path.join(folder,  rf'MRI_nii_registered/{case}/DWI_seg_registered_corrected.nii.gz')
    # load image data
    image_data_dict = read_nii_folder(ctp_folder)
    dwi_segmentation = sitk.ReadImage(dwi_segmentation)
    # load time matrix
    time_data = np.load(time_matrix)
    # load aif and vof locations
    aif_location = sitk.GetArrayFromImage(sitk.ReadImage(aif_location))
    vof_location = sitk.GetArrayFromImage(sitk.ReadImage(vof_location))
    time_aif_location = list(set(np.where(aif_location == 1)[0]))[0]
    time_vof_location = list(set(np.where(vof_location == 1)[0]))[0]
    # load brainmask
    brainmask_data = sitk.GetArrayFromImage(sitk.ReadImage(brainmask))
    # get aif and vof data
    aif_data = np.sum(np.multiply(aif_location, image_data_dict['array']), axis=(1,2,3)) / np.sum(aif_location)
    vof_data = np.sum(np.multiply(vof_location, image_data_dict['array']), axis=(1,2,3)) / np.sum(vof_location)
    aif_time_data = time_data[time_aif_location]
    vof_time_data = time_data[time_vof_location]

    # scale aif
    vof_baseline = np.mean(vof_data[:4])
    aif_baseline = np.mean(aif_data[:4])
    aif_wo_baseline = aif_data - aif_baseline
    aif_part_nonzero_baseline = aif_wo_baseline.clip(max=0)
    aif_wo_baseline = aif_wo_baseline.clip(min=0)
    vof_wo_baseline = vof_data - vof_baseline
    # now we use simpsons approximation because of irregular timing
    cumsum_aif = simpson(aif_wo_baseline, aif_time_data)
    cumsum_vof = simpson(vof_wo_baseline, vof_time_data)
    # cumsum_aif = np.cumsum(aif_wo_baseline)[-1]
    # cumsum_vof = np.cumsum(vof_wo_baseline)[-1]
    ratio = cumsum_vof / cumsum_aif
    aif_data = aif_wo_baseline * ratio + aif_baseline
    aif_data += aif_part_nonzero_baseline * ratio

    image_data_dict['array'] = np.multiply(image_data_dict['array'], brainmask_data)
    image_data_dict['mip'] = np.max(image_data_dict['array'], axis=0)

    vesselmask = np.zeros_like(image_data_dict['mip'])
    vesselmask[image_data_dict['mip']>150] = 1

    tissuemask = np.zeros_like(image_data_dict['mip'])
    image_data_avg = np.mean(image_data_dict['array'][:4,...], axis=0)
    tissuemask[(image_data_avg > 30) & (image_data_avg < 100)] = 1

    complete_mask = np.zeros_like(image_data_dict['mip'])
    valid_voxels = np.where((brainmask_data == 1) & (tissuemask == 1) & (vesselmask == 0))

    complete_mask[valid_voxels] = 1
    # If smoothing, apply here
    if gaussian_filter_type:
        image_data_dict['array'] = apply_gaussian_filter_with_mask(gaussian_filter_type,
                                                                   image_data_dict['array'].copy(),
                                                                   complete_mask,
                                                                   sd=sd)
    image_data_dict['array'] = image_data_dict['array'].astype(np.float32)
    image_data_dict['array'] = rearrange(image_data_dict['array'], 't d h w -> d h w t')


    data_dict = {'aif': aif_data,
                 'vof': vof_data,
                 'time': time_data,
                 'curves': image_data_dict['array'],
                 'brainmask': brainmask_data,
                 'mip': image_data_dict['mip'],
                 'mask': complete_mask,
                 }


    # create meshes
    data_dict = get_coll_points(data_dict)
    data_dict = normalize_data(data_dict)
    data_dict['time'] = np.tile(
        data_dict['time'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, 512, 512, 1),
    ).astype(np.float32)
    data_dict['time_inference_highres'] = np.tile(
        data_dict['time_inference_highres'][..., np.newaxis, np.newaxis, np.newaxis],
        (1, 512, 512, 1),
    ).astype(np.float32)
    data_dict = create_mesh_amc(data_dict)
    data_dict = get_tensors(data_dict)
    data_dict['aif_time'] = data_dict['time'][time_aif_location]
    data_dict['dwi_segmentation'] = dwi_segmentation
    return data_dict

def create_mesh_amc(data_dict):
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0, data_dict['time'].shape[2] - 1, num=data_dict['time'].shape[2]),
        np.linspace(0, data_dict['time'].shape[3] - 1, num=data_dict['time'].shape[3]),
    )
    mesh_data = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (data_dict['time'].shape[0],data_dict['time'].shape[1], 1, 1, 1)
    ).astype(np.float32)

    mesh_data_xy = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (1, 1, 1)
    ).astype(np.float32)

    mesh_data_hr = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (data_dict['time_inference_highres'].shape[0], 1, 1, 1)
    ).astype(np.float32)

    data_dict['mesh_mean'] = mesh_data.mean()
    data_dict['mesh_std'] = mesh_data.std()
    data_dict['mesh_max'] = mesh_data.max()
    data_dict['mesh_min'] = mesh_data.min()

    mesh_data_hr = (mesh_data_hr - mesh_data.mean()) / mesh_data.std()
    mesh_data_xy = (mesh_data_xy - mesh_data.mean()) / mesh_data.std()
    mesh_data = (mesh_data - mesh_data.mean()) / mesh_data.std()

    data_dict['coordinates_highres'] = np.concatenate([data_dict['time_inference_highres'], mesh_data_hr], axis=3)

    data_dict['coordinates'] = np.concatenate([data_dict['time'], mesh_data], axis=4)
    data_dict['coordinates_xy_only'] = mesh_data_xy[np.newaxis, np.newaxis, ...]
    data_dict['coordinates'] = rearrange(data_dict['coordinates'], 'dim1 t x y vals -> dim1 x y t vals')
    data_dict['coordinates_highres'] = rearrange(data_dict['coordinates_highres'],'t x y vals -> x y t vals')

    return data_dict
def create_mesh(data_dict):
    mesh_x, mesh_y = np.meshgrid(
        np.linspace(0, data_dict['time'].shape[1] - 1, num=data_dict['time'].shape[1]),
        np.linspace(0, data_dict['time'].shape[2] - 1, num=data_dict['time'].shape[2]),
    )
    mesh_data = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (len(data_dict['time']), 1, 1, 1)
    ).astype(np.float32)

    mesh_data_hr = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (len(data_dict['time_inference_highres']), 1, 1,1)
    ).astype(np.float32)

    mesh_data_xy = np.tile(
        np.stack((mesh_x, mesh_y), axis=-1), (1, 1, 1)
    ).astype(np.float32)

    data_dict['mesh_mean'] = mesh_data.mean()
    data_dict['mesh_std'] = mesh_data.std()
    data_dict['mesh_max'] = mesh_data.max()
    data_dict['mesh_min'] = mesh_data.min()

    mesh_data_hr = (mesh_data_hr - mesh_data.mean()) / mesh_data.std()
    mesh_data_xy = (mesh_data_xy - mesh_data.mean()) / mesh_data.std()
    mesh_data = (mesh_data - mesh_data.mean()) / mesh_data.std()

    data_dict['coordinates'] = np.concatenate([data_dict['time'], mesh_data], axis=3)[np.newaxis, np.newaxis, ...]
    data_dict['coordinates_xy_only'] = mesh_data_xy[np.newaxis, np.newaxis, ...]
    data_dict['time_xy_highres'] = np.concatenate([data_dict['time_inference_highres'], mesh_data_hr], axis=3)[np.newaxis, np.newaxis, ...]
    data_dict['coordinates'] = rearrange(data_dict['coordinates'], 'dim1 dim2 t x y vals -> dim1 dim2 x y t vals')
    data_dict['time_xy_highres'] = rearrange(data_dict['time_xy_highres'],'dim1 dim2 t x y vals -> dim1 dim2 x y t vals')

    return data_dict

def read_nii_folder(folder):
    scans = sorted(glob.glob(folder))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(scans)
    image = reader.Execute()
    data_dict = {'array': sitk.GetArrayFromImage(image).astype(np.float32),
                 'spacing': image.GetSpacing(),
                 'dims': image.GetSize(),
                 'shape': sitk.GetArrayFromImage(image).shape}
    return data_dict

def normalize_data(data_dict):
    # input normalization
    data_dict['std_t'] = data_dict['time'].std()
    data_dict['mean_t'] = data_dict['time'].mean()
    data_dict['time'] = (data_dict['time'] - data_dict['time'].mean()) / data_dict['time'].std()
    data_dict['time_inference_highres'] = np.array([float(x) for x in np.arange(np.min(data_dict['time']), np.max(data_dict['time']) + 0.06, 0.06)])
    # output normalization
    max_ = data_dict['aif'].max()
    data_dict['aif'] /= max_
    data_dict['vof'] /= max_
    data_dict['curves'] /= max_
    return data_dict


def get_coll_points(data_dict):
    data_dict['coll_points'] = np.random.uniform(
        np.min(data_dict['time']), np.max(data_dict['time']), len(data_dict['time']) * 5 * 10 * 3
    ).astype(np.float32)
    data_dict['bound'] = np.array([np.min(data_dict['time'])])
    data_dict['coll_points_max'] = np.max(data_dict['coll_points'])
    data_dict['coll_points_min'] = np.min(data_dict['coll_points'])

    return data_dict


def get_tensors(data_dict):
    for key in data_dict.keys():
        data_dict[key] = torch.as_tensor(data_dict[key], dtype=torch.float32)
    return data_dict

# def undersample(data_dict, degree):
#     assert degree in [0, 0.50, 0.25]
#     if degree == 0:
#         return data_dict
#     frames = len(data_dict['aif'])
#     if degree == 0.50:
#         indices = np.arange(0, frames, 2)
#     elif degree == 0.25:
#         indices = np.arange(0, frames, 4)
#     else:
#         raise NotImplementedError('Amount of undersampling not implemented')
#     data_dict['aif'] = data_dict['aif'][indices]
#     data_dict['vof'] = data_dict['vof'][indices]
#     data_dict['time'] = data_dict['time'][indices]
#     data_dict['curves'] = data_dict['curves'][...,indices]
#     return data_dict
def undersample(image_data, time, degree):
    assert degree in [0, 0.50, 0.25]
    if degree == 0:
        return image_data, time
    frames = len(time)
    if degree == 0.50:
        indices = np.arange(0, frames, 2)
    elif degree == 0.25:
        indices = np.arange(0, frames, 4)
    else:
        raise NotImplementedError('Amount of undersampling not implemented')
    image_data = image_data[:,indices,...]
    time = time[indices]
    return image_data, time

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
    mask = np.expand_dims(mask, 0)
    mask = np.repeat(mask, array.shape[0], axis=0)
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
    plt.imshow(array[4, 0], vmin=0, vmax=100)
    plt.show()
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            filtered[i, j] = denoise_bilateral(array[i, j], win_size=None,
                                               sigma_color=20, sigma_spatial=sigma_spatial,
                                               bins=10000)
    filtered *= mask
    plt.imshow(filtered[4, 0], vmin=0, vmax=100)
    plt.show()
    return filtered

def save_perfusion_parameters_amc(config, case, cbf_results, cbv_results, mtt_results, delay_results, tmax_results, data_dict):
    dwi_segmentation = os.path.join(r'D:/PPINN_patient_data/AMCCTP', rf'MRI_nii_registered/{case}/DWI_seg_registered_corrected.nii.gz')
    template = sitk.ReadImage(dwi_segmentation)
    sitk.WriteImage(template, os.path.join(wandb.run.dir, 'results', case, 'dwi_seg.nii'))

    brainmask = sitk.ReadImage(os.path.join(r'D:/PPINN_patient_data/AMCCTP', rf'CTP_nii_brainmask/{case}/brainmask.nii.gz'))
    sitk.WriteImage(brainmask, os.path.join(wandb.run.dir, 'results', case, 'brainmask.nii'))

    mask = np2itk(data_dict['mask'], template)
    sitk.WriteImage(mask, os.path.join(wandb.run.dir, 'results', case, 'vesselmask.nii'))

    cbf_results = np2itk(cbf_results, template)
    sitk.WriteImage(cbf_results, os.path.join(wandb.run.dir, 'results', case, 'cbf.nii'))

    cbv_results = np2itk(cbv_results, template)
    sitk.WriteImage(cbv_results, os.path.join(wandb.run.dir, 'results', case, 'cbv.nii'))

    mtt_results = np2itk(mtt_results, template)
    sitk.WriteImage(mtt_results, os.path.join(wandb.run.dir, 'results', case, 'mtt.nii'))

    delay_results = np2itk(delay_results, template)
    sitk.WriteImage(delay_results, os.path.join(wandb.run.dir, 'results', case, 'delay.nii'))

    tmax_results = np2itk(tmax_results, template)
    sitk.WriteImage(tmax_results, os.path.join(wandb.run.dir, 'results', case, 'tmax.nii'))

def save_perfusion_parameters(config, case, cbf_results, cbv_results, mtt_results, delay_results, tmax_results):
    mode = 'TRAINING' if config.mode == 'train' else 'TESTING'
    folder = r'data/ISLES2018'

    template_file = glob.glob(os.path.join(folder, mode, case, '*OT*', '*OT*.nii'))[0]
    template = sitk.ReadImage(template_file)
    sitk.WriteImage(template, os.path.join(wandb.run.dir, 'results', case, 'dwi_seg.nii'))

    cbf_results = np2itk(cbf_results, template)
    sitk.WriteImage(cbf_results, os.path.join(wandb.run.dir, 'results', case, 'cbf.nii'))

    cbv_results = np2itk(cbv_results, template)
    sitk.WriteImage(cbv_results, os.path.join(wandb.run.dir, 'results', case, 'cbv.nii'))

    mtt_results = np2itk(mtt_results, template)
    sitk.WriteImage(mtt_results, os.path.join(wandb.run.dir, 'results', case, 'mtt.nii'))

    delay_results = np2itk(delay_results, template)
    sitk.WriteImage(delay_results, os.path.join(wandb.run.dir, 'results', case, 'delay.nii'))

    tmax_results = np2itk(tmax_results, template)
    sitk.WriteImage(tmax_results, os.path.join(wandb.run.dir, 'results', case, 'tmax.nii'))

def np2itk(arr, original_img):
    img = sitk.GetImageFromArray(arr, False)
    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    img.CopyInformation(original_img)
    return img

class CurveDataset(Dataset):
    def __init__(self, data_curves, data_coordinates, collo_coordinates):
        self.data_curves = data_curves
        self.data_coordinates = data_coordinates
        self.collo_coordinates = collo_coordinates
    def __len__(self):
        return len(self.data_curves)
    def __getitem__(self, idx):
        return self.data_curves[idx], self.data_coordinates[idx], self.collo_coordinates[idx]