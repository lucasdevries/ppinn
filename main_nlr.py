from utils import data_utils
from scipy.ndimage import convolve
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
def fun(x, auc, tac_k):
    tac_est = gen_tac(x[0]/x[1], x[1], x[2], auc)
    sse = np.sum((tac_est-tac_k)**2)
    return sse

def gen_tac(cbf, mtt, d, auc):
    n = len(auc)
    ns = np.arange(n)
    interpolator_a = interp1d(ns, auc, kind='linear', bounds_error=False, fill_value=0)
    interpolator_b = interp1d(ns, auc, kind='linear', bounds_error=False, fill_value=0)
    a = interpolator_a([i - 0.5 - d for i in ns])
    b = interpolator_b([i - 0.5 - d - mtt for i in ns])
    tac = cbf * (a - b)
    return tac

def boxnlr(aif, tac, dt):
    result = sitk.ReadImage(r'L:\basic\divi\CMAJOIE\CLEOPATRA\Substudies\Lucas\KudoPhantom\unfiltered_rescaled_aif.nii')
    result = sitk.GetArrayFromImage(result)
    result = result[...,144:-144, 144:-144]
    density = 1.05
    constant = (100 / density) * 0.55 / 0.75
    cbv = result[0]
    cbv = cbv * constant
    mtt = result[1]
    cbf = cbv / (mtt / 60)
    delay = result[2]
    plt.hist(mtt.flatten(), bins=50)
    plt.show()
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(np.transpose(cbv), vmin=0.0, vmax=7, cmap='jet')
    ax[1].imshow(np.transpose(cbf), vmin=0, vmax=90, cmap='jet')
    ax[2].imshow(np.transpose(mtt), vmin=0, vmax=24, cmap='jet')
    ax[3].imshow(np.transpose(delay), vmin=0, vmax=3.5, cmap='jet')
    plt.show()

    x0 = [0.05, 4/dt, 1/dt]
    k = [0.25, 0.5, 0.25]
    aif_k = convolve(aif, k, mode='nearest')
    tac_k = convolve(tac, k, mode='nearest')
    auc = np.cumsum(aif_k)

    # here minimize
    x = minimize(fun, x0, args=(auc, tac_k), method='Nelder-Mead').x
    x = [x[0], x[1]*dt, x[2]*dt]
    return x

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
            x = boxnlr(aif, tac[0,0,i,j], dt=2)
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