import os
from utils import data_utils, train_utils
from utils.config import process_config
import wandb
import matplotlib.pyplot as plt
from models.ppinn_models import PPINN
import torch
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
    os.environ["WANDB_MODE"] = "online"
    # Pass them to wandb.init
    wandb.init(config=dict(config), project="ppinn")
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    train_utils.set_seed(config['seed'])
    config['run_name'] = wandb.run.name
    config['run_id'] = wandb.run.id
    train(config)

def train(config):
    data_dict = data_utils.load_data(gaussian_filter_type=config.gaussian_filter_type, sd=config.sd,
                                     cbv_slice=config.cbv_slice, simulation_method=config.simulation_method)
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)
    ppinn = PPINN(config,
                  shape_in=shape_in,
                  perfusion_values=data_dict['perfusion_values'],
                  n_inputs=1,
                  std_t=data_dict['std_t'])

    ppinn.plot_params(0,0, perfusion_values=data_dict['perfusion_values'], epoch=0)
    ppinn.fit(data_dict['time'],
              data_dict['aif'],
              data_dict['curves'],
              data_dict['coll_points'],
              data_dict['bound'],
              data_dict['perfusion_values'],
              batch_size=config.batch_size,
              epochs=config.epochs)
    ppinn.save_parameters()
if __name__ == "__main__":
    main()