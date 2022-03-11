from utils import data_utils
import matplotlib.pyplot as plt
from models.ppinn_models import PPINN
from torch.utils.data import DataLoader

def train():
    data_dict = data_utils.load_data(gaussian_filter_type='spatio-temporal')
    print(data_dict['aif'])
    print(type(data_dict['perfusion_values']))
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)

    train_dataloader = DataLoader(data_dict['time'], batch_size=10, shuffle=False)
    train_features = next(iter(train_dataloader))
    print(train_features)
    train_dataloader = DataLoader(data_dict['time'], batch_size=10, shuffle=False)
    for data in train_dataloader:
        print(data)
    ppinn = PPINN(shape_in=shape_in,
                               n_layers=3,
                               n_units=32,
                               lr=1e-3,
                               loss_weights=(1, 1, 1),
                               bn=False,
                               trainable_params='all',
                               n_inputs=1,
                               std_t=data_dict['std_t'])



if __name__ == "__main__":
    train()
