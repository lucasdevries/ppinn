from utils import data_utils
import matplotlib.pyplot as plt
from models.ppinn_models import PPINN
from torch.utils.data import DataLoader

def train():
    data_dict = data_utils.load_data(gaussian_filter_type='spatial', sd=2.5)

    for i in range(0,128):
        plt.plot(data_dict['time'], data_dict['curves'][0,0,i,0,:].numpy())
    # plt.ylim(0.05, 0.3)
    plt.show()
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)
    ppinn = PPINN(shape_in=shape_in,
                               n_layers=2,
                               n_units=16,
                               lr=1e-3,
                               perfusion_values=data_dict['perfusion_values'],
                               loss_weights=(1, 10, 0),
                               bn=False,
                               trainable_params='all',
                               n_inputs=1,
                               std_t=data_dict['std_t'])

    # ppinn.plot_params(0,0, perfusion_values=data_dict['perfusion_values'], epoch='Start')
    ppinn.fit(data_dict['time'],
              data_dict['aif'],
              data_dict['curves'],
              data_dict['coll_points'],
              data_dict['bound'],
              data_dict['perfusion_values'],
              batch_size=32,
              epochs=5000)

    ppinn.plot_params(0,0, perfusion_values=data_dict['perfusion_values'], epoch='End')
    # print(data_dict['perfusion_values'])
if __name__ == "__main__":
    train()
