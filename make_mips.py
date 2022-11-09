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
from utils.data_utils import read_nii_folder, apply_gaussian_filter, np2itk

def make_mip(case, gaussian_filter_type='gauss_spatial'):
    folder = r"D:\PPINN_patient_data\AMCCTP"
    ctp_folder = os.path.join(folder, rf'CTP_nii_registered/{case}/*.nii.gz')
    brainmask = os.path.join(folder, rf'CTP_nii_brainmask/{case}/brainmask.nii.gz')
    # load image data
    image_data_dict = read_nii_folder(ctp_folder)
    # load brainmask
    brainmask_data = sitk.GetArrayFromImage(sitk.ReadImage(brainmask))
    image_data_dict['array'] = np.multiply(image_data_dict['array'], brainmask_data)
    image_data_dict['array'] = image_data_dict['array'].astype(np.float32)
    image_data_dict['array'] = rearrange(image_data_dict['array'], 't d h w -> d h w t')

    mip = np.max(image_data_dict['array'], axis=3)
    mip_sitk = np2itk(mip, sitk.ReadImage(brainmask))
    os.makedirs(os.path.join(folder, 'MIP',case), exist_ok=True)
    sitk.WriteImage(mip_sitk, os.path.join(folder, 'MIP',case,'mip.nii.gz'))

if __name__ == '__main__':
    folder = r"D:\PPINN_patient_data\AMCCTP"
    print(os.listdir(os.path.join(folder, rf'CTP_nii_registered')))
    for i in os.listdir(os.path.join(folder, rf'CTP_nii_registered')):

        make_mip(i)