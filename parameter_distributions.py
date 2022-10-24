#%%

import glob, os

from tqdm import tqdm
import SimpleITK as sitk

import matplotlib.pyplot as plt

#%%

from utils.val_utils import drop_unphysical_amc
amc_dir = r'data/results_ppinn_amcctp'
isles_dir = r'data/results_ppinn_isles_upsampled'


#%%

cases = os.listdir(isles_dir)
values_isles = {}
values_isles['cbf'] = []
values_isles['cbv'] = []
values_isles['mtt'] = []
values_isles['delay'] = []
values_isles['tmax'] = []

for case in tqdm(cases):
    data = {}
    data['cbf'] = sitk.ReadImage(os.path.join(isles_dir,case,'cbf.nii'))
    data['cbv'] = sitk.ReadImage(os.path.join(isles_dir,case,'cbv.nii'))
    data['mtt' ]= sitk.ReadImage(os.path.join(isles_dir,case,'mtt.nii'))
    data['delay'] = sitk.ReadImage(os.path.join(isles_dir,case,'delay.nii'))
    data['tmax'] = sitk.ReadImage(os.path.join(isles_dir,case,'tmax.nii'))

    for key, val in data.items():
        value_list = sitk.GetArrayFromImage(val).flatten().tolist()
        values_isles[key].append(value_list)

for key, val in values_isles.items():
    values_isles[key] = [item for sublist in val for item in sublist]

#%%
cases = os.listdir(amc_dir)
values_amc = {}
values_amc['cbf'] = []
values_amc['cbv'] = []
values_amc['mtt'] = []
values_amc['delay'] = []
values_amc['tmax'] = []

for case in tqdm(cases):
    data = {}
    data['cbf'] = sitk.ReadImage(os.path.join(amc_dir,case,'cbf.nii'))
    data['cbv'] = sitk.ReadImage(os.path.join(amc_dir,case,'cbv.nii'))
    data['mtt' ]= sitk.ReadImage(os.path.join(amc_dir,case,'mtt.nii'))
    data['delay'] = sitk.ReadImage(os.path.join(amc_dir,case,'delay.nii'))
    data['tmax'] = sitk.ReadImage(os.path.join(amc_dir,case,'tmax.nii'))

    for key, val in data.items():
        value_list = sitk.GetArrayFromImage(val).flatten().tolist()
        values_amc[key].append(value_list)

for key, val in values_amc.items():
    values_amc[key] = [item for sublist in val for item in sublist]

#%%

len(values_amc['cbf']), len(values_isles['cbf'])


#%%
fig, ax = plt.subplots(1,2)
ax[0].hist(values_isles['cbf'], range=(1,150),density=True, bins=20)
ax[1].hist(values_amc['cbf'], range=(1,150),density=True, bins=20)
plt.savefig('cbf.png')
plt.close()

#%%

fig, ax = plt.subplots(1,2)
ax[0].hist(values_isles['cbv'], range=(0,20),density=True, bins=20)
ax[1].hist(values_amc['cbv'], range=(0,20),density=True, bins=20)
plt.savefig('cbv.png')

plt.close()

#%%

fig, ax = plt.subplots(1,2)
ax[0].hist(values_isles['mtt'], range=(0,30),density=True, bins=20)
ax[1].hist(values_amc['mtt'], range=(0,30),density=True, bins=20)
plt.savefig('mtt.png')

plt.close()

#%%

fig, ax = plt.subplots(1,2)
ax[0].hist(values_isles['delay'], range=(0,10),density=True, bins=20)
ax[1].hist(values_amc['delay'], range=(0,10),density=True, bins=20)
plt.savefig('delay.png')

plt.close()

#%%

fig, ax = plt.subplots(1,2)
ax[0].hist(values_isles['tmax'], range=(0,15),density=True, bins=20)
ax[1].hist(values_amc['tmax'], range=(0,15),density=True, bins=20)
plt.savefig('tmax.png')
plt.close()

#%%


