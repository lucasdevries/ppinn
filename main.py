from utils import data_utils
import matplotlib.pyplot as plt
from models.ppinn_models import PPINN
from torch.utils.data import DataLoader
import torch
def train():
    data_dict = data_utils.load_data(gaussian_filter_type='spatial', sd=2.5)
    # plt.plot(data_dict['time'], data_dict['aif'], c='k')
    # for i in range(208, 209):
    #     for j in range(16, 208, 32):
    #         plt.scatter(data_dict['time'], data_dict['curves'][0,0,j,i,:].numpy())
    # plt.show()
    # for i in range(16, 208, 32):
    #     for j in range(16, 208, 32):
    #         get_bolus_arrival_time(data_dict['curves'][0,0,i,j,:])
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)
    ppinn = PPINN(shape_in=shape_in,
                               n_layers=2,
                               n_units=16,
                               lr=1e-2,
                               perfusion_values=data_dict['perfusion_values'],
                               loss_weights=(1, 10, 0),
                               bn=False,
                               trainable_params='all',
                               n_inputs=1,
                               std_t=data_dict['std_t'])

    ppinn.plot_params(0,0, perfusion_values=data_dict['perfusion_values'], epoch='Start')
    ppinn.fit(data_dict['time'],
              data_dict['aif'],
              data_dict['curves'],
              data_dict['coll_points'],
              data_dict['bound'],
              data_dict['perfusion_values'],
              batch_size=32,
              epochs=6000)

    ppinn.plot_params(0,0, perfusion_values=data_dict['perfusion_values'], epoch='End')
    # print(data_dict['perfusion_values'])


def get_bolus_arrival_time(curve):

    curve_dt = torch.gradient(curve)[0]
    curve_dtdt = torch.gradient(curve_dt)[0]
    top_2_maximums = torch.topk(curve_dtdt, k=2)
    index_of_bolus_arrival = torch.min(top_2_maximums.indices).item()

    plt.plot(curve.numpy(), label='curve')
    # plt.plot(10 * curve_dt.numpy(), label='first derivative')
    # plt.plot(10 * curve_dtdt.numpy(), label='second derivative')
    plt.scatter(index_of_bolus_arrival, curve.numpy()[index_of_bolus_arrival], label='Bolus arrival')
    # plt.legend()
    plt.show()
if __name__ == "__main__":
    train()
