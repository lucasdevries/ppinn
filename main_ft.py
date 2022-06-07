from utils import data_utils
from scipy.ndimage import convolve
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk


def train():
    data_dict = data_utils.load_data(gaussian_filter_type='spatial', sd=2.5,
                                     cbv_slice=4, simulation_method=2, method='nlr')
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)

    x, y = 224, 224
    cbv = np.zeros((x,y))
    mtt = np.zeros((x,y))
    delay = np.zeros((x,y))

    aif = data_dict['aif']
    tac = data_dict['curves']
    for i in tqdm(range(x)):
        for j in range(y):

            print(list(tac[0, 0, 200, 200]))






            cbv[i,j] = x[0]
            mtt[i, j] = x[1]
            delay[i, j] = x[2]

    density = 1.05
    constant = (100 / density) * 0.55 / 0.75
    cbv = cbv * constant
    cbf = cbv / (mtt / 60)

    fig, ax = plt.subplots(1,4)
    ax[0].imshow(cbv, vmin=0, vmax=50, cmap='jet')
    ax[1].imshow(mtt, vmin=0, vmax=30, cmap='jet')
    ax[2].imshow(delay, vmin=-3, vmax=3, cmap='jet')
    ax[3].imshow(cbf, vmin=0, vmax=200, cmap='jet')
    plt.show()
    fig, ax = plt.subplots(1,4)
    ax[0].hist(cbv.flatten())
    ax[1].hist(mtt.flatten())
    ax[2].hist(delay.flatten())
    ax[3].hist(cbf.flatten())
    plt.show()
if __name__ == "__main__":
    train()