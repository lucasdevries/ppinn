from utils import data_utils
import matplotlib.pyplot as plt
from models.ppinn_models import PPINN
from torch.utils.data import DataLoader

def train():
    data_dict = data_utils.load_data(gaussian_filter_type='spatial')
    for i in range(30):
        plt.plot(data_dict['curves'][0,0,0,i,:].numpy())
    plt.ylim(0.05, 0.3)
    plt.show()
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)
    ppinn = PPINN(shape_in=shape_in,
                               n_layers=2,
                               n_units=16,
                               lr=1e-3,
                               loss_weights=(1, 100, 0),
                               bn=False,
                               trainable_params='all',
                               n_inputs=1,
                               std_t=data_dict['std_t'])

    cbf = data_dict['perfusion_values'][...,-1:]
    ppinn.plot_params(0,0, perfusion_values=cbf, epoch='Start')
    ppinn.fit(data_dict['time'],
              data_dict['aif'],
              data_dict['curves'],
              data_dict['coll_points'],
              data_dict['bound'],
              data_dict['perfusion_values'][..., -1:],
              batch_size=32,
              epochs=60000)


    # ppinn.plot_params(0,0, perfusion_values=cbf, epoch='End')
    # print(data_dict['perfusion_values'])
if __name__ == "__main__":
    train()
