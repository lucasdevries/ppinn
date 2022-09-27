import os
from utils import data_utils, train_utils
from utils.config import process_config
import wandb
from models.ppinn_models import PPINN
from models.ppinn_model_isles import PPINN_isles
import numpy as np
import argparse
from tqdm import tqdm
from utils.val_utils import visualize, load_sygnovia_results, load_nlr_results, load_phantom_gt
from utils.val_utils import log_software_results, plot_results, drop_edges, drop_unphysical
import matplotlib.pyplot as plt
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
    os.makedirs(os.path.join(wandb.run.dir, 'results'))

    results = {
        'gt': load_phantom_gt(cbv_ml=config.cbv_ml),
        'sygnovia': load_sygnovia_results(cbv_ml=config.cbv_ml, sd=config.sd),
        'nlr': load_nlr_results(cbv_ml=config.cbv_ml, sd=config.sd),
        'ppinn': train(config)
    }

    plot_results(results)
    log_software_results(results, config.cbv_ml)
    results = drop_edges(results) # if config.drop_edges else results
    results = drop_unphysical(results) # if config.drop_unphysical else results
    plot_results(results, corrected=True)
    log_software_results(results, config.cbv_ml, corrected=True)

def train(config):
    data_dict = data_utils.load_data(gaussian_filter_type=config.filter_type,
                                     sd=config.sd,
                                     cbv_ml=config.cbv_ml,
                                     simulation_method=config.simulation_method,
                                     temporal_smoothing=config.temporal_smoothing,
                                     baseline_zero=config.baseline_zero)
    shape_in = data_dict['perfusion_values'].shape[:-1]  # (3, 5, 224, 224)
    ppinn = PPINN(config,
                  shape_in=shape_in,
                  perfusion_values=data_dict['perfusion_values'],
                  n_inputs=1,
                  std_t=data_dict['std_t'])
    # ppinn.plot_params(0,0, perfusion_values=data_dict['perfusion_values'], epoch=0)
    # ppinn.plot_params_difference(0,0, perfusion_values=data_dict['perfusion_values'], epoch=0)

    ppinn.fit(data_dict['time'],
              data_dict['aif'],
              data_dict['curves'],
              data_dict['coll_points'],
              data_dict['bound'],
              data_dict['perfusion_values'],
              batch_size=config.batch_size,
              epochs=config.epochs)
    ppinn.save_parameters()
    ppinn_results = ppinn.get_results()
    return ppinn_results
def train_isles(config):
    folder = 'TRAINING' if config.mode == 'train' else 'TESTING'
    cases = os.listdir(r'data/ISLES2018/{}'.format(folder))
    for case in cases[10:]:
        os.makedirs(os.path.join(wandb.run.dir, 'results',case))
        data_dict = data_utils.load_data_ISLES(filter_type=config.filter_type,
                                               sd=config.sd,
                                               temporal_smoothing=config.temporal_smoothing,
                                               baseline_zero=config.baseline_zero,
                                               mode=config.mode,
                                               case=case)
        shape_in = data_dict['curves'][0:1].shape[:-1]
        scan_dimensions = data_dict['curves'].shape[:-1]
        slices = scan_dimensions[0]
        cbf_results = np.zeros([*scan_dimensions], dtype=np.float32)
        cbv_results = np.zeros([*scan_dimensions], dtype=np.float32)
        mtt_results = np.zeros([*scan_dimensions], dtype=np.float32)
        delay_results = np.zeros([*scan_dimensions], dtype=np.float32)
        tmax_results = np.zeros([*scan_dimensions], dtype=np.float32)

        for slice in tqdm(range(slices)):
            ppinn = PPINN_isles(config,
                                shape_in=shape_in,
                                perfusion_values=data_dict['perfusion_values'],
                                n_inputs=1,
                                std_t=data_dict['std_t'])
            result_dict = ppinn.fit(slice,
                      data_dict,
                      batch_size=config.batch_size,
                      epochs=int(config.epochs))
            cbf_results[slice, ...] = result_dict['cbf'].cpu().detach().numpy()
            cbv_results[slice, ...] = result_dict['cbv'].cpu().detach().numpy()
            mtt_results[slice, ...] = result_dict['mtt'].cpu().detach().numpy()
            delay_results[slice, ...] = result_dict['delay'].cpu().detach().numpy()
            tmax_results[slice, ...] = result_dict['tmax'].cpu().detach().numpy()

            visualize(slice, case, data_dict['perfusion_values'][slice], result_dict)

        data_utils.save_perfusion_parameters(config,
                                             case,
                                             cbf_results,
                                             cbv_results,
                                             mtt_results,
                                             delay_results,
                                             tmax_results)

if __name__ == "__main__":
    main()