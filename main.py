import os
from utils import data_utils, train_utils
from utils.config import process_config
import wandb
from models.ppinn_models import PPINN
from models.ppinn_model_isles import PPINN_isles

import argparse
def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    # parse the config json file
    config = process_config(args.config)
    # set environment variable for offline runs
    os.environ["WANDB_MODE"] = "online" if config.wandb else "offline"
    # Pass them to wandb.init
    wandb.init(config=dict(config), project="ppinn")
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    train_utils.set_seed(config['seed'])
    config['run_name'] = wandb.run.name
    config['run_id'] = wandb.run.id
    train_isles(config)

def train(config):
    data_dict = data_utils.load_data(gaussian_filter_type=config.filter_type,
                                     sd=config.sd,
                                     cbv_slice=config.cbv_slice,
                                     simulation_method=config.simulation_method,
                                     temporal_smoothing=config.temporal_smoothing,
                                     baseline_zero=config.baseline_zero)
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)
    ppinn = PPINN(config,
                  shape_in=shape_in,
                  perfusion_values=data_dict['perfusion_values'],
                  n_inputs=1,
                  std_t=data_dict['std_t'])

    ppinn.plot_params(0,0, perfusion_values=data_dict['perfusion_values'], epoch=0)
    ppinn.plot_params_difference(0,0, perfusion_values=data_dict['perfusion_values'], epoch=0)

    ppinn.fit(data_dict['time'],
              data_dict['aif'],
              data_dict['curves'],
              data_dict['coll_points'],
              data_dict['bound'],
              data_dict['perfusion_values'],
              batch_size=config.batch_size,
              epochs=config.epochs)
    ppinn.save_parameters()

def train_isles(config):
    data_dict = data_utils.load_data_ISLES(filter_type=config.filter_type,
                                     sd=config.sd,
                                     temporal_smoothing=config.temporal_smoothing,
                                     baseline_zero=config.baseline_zero)
    slice = 6
    shape_in = data_dict['curves'][slice:slice+1].shape[:-1]
    ppinn = PPINN_isles(config,
                  shape_in=shape_in,
                  perfusion_values=data_dict['perfusion_values'],
                  n_inputs=1,
                  std_t=data_dict['std_t'])

    ppinn.plot_params(slice=slice, epoch=0, brainmask=data_dict['brainmask'])



    ppinn.fit(slice,
              data_dict['time'],
              data_dict['aif'],
              data_dict['curves'],
              data_dict['coll_points'],
              data_dict['bound'],
              batch_size=config.batch_size,
              epochs=config.epochs,
              brainmask=data_dict['brainmask'])
    ppinn.save_parameters()
if __name__ == "__main__":
    main()