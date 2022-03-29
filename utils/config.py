import os
import json
from easydict import EasyDict
from pprint import pprint

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
        except ValueError as e:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)
    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(json_file):
    """
    Get the json file then editing the path of the experiments folder, creating the dir and return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(json_file)
    print(" THE Configuration of your experiment ..")
    print(" *************************************** ")
    return config