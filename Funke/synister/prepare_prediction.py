import os
from copy import deepcopy
import sys
from shutil import copyfile, rmtree
import json
import configargparse
import configparser
from os.path import expanduser
import click
import numpy as np
from synister.read_config import read_train_config, read_predict_config


p = configargparse.ArgParser()
p.add('-d', '--base_dir', required=True, help='base directory for storing synister experiments')
p.add('-e', required=True, help='name of the experiment, e.g. fafb, defaults to ``base``')
p.add('-t', required=True, help='train setup number to use for this prediction')
p.add('-i', nargs="+",required=True, help='iterations to use for validation')
p.add('-v', required=False, action='store_true', help='use validation split part')
p.add('-c', required=False, action='store_true', help='clean up - remove specified predict setup')
this_dir = os.path.dirname(__file__)

def set_up_environments(base_dir,
                        experiment,
                        train_number,
                        iterations,
                        clean_up,
                        validation):

    #predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "03_predict/setup_t{}_p{}".format(train_number, predict_number))
    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "03_predict")

    # Predict runs of the same train run that already exist:
    existing_predicts = [os.path.join(predict_setup_dir, f) for f in os.listdir(predict_setup_dir) if "setup_t{}".format(train_number) in f]
    existing_predict_configs = [read_predict_config(os.path.join(f, "predict_config.ini")) for f in existing_predicts]
    existing_predict_numbers = [cfg["predict_number"] for cfg in existing_predict_configs]

    # Check for overlapping already existing predicts:
    collisions = {i: None for i in iterations}
    for cfg in existing_predict_configs:
        i_cfg = int(cfg["train_checkpoint"].split("_")[-1])
        val_cfg = cfg["split_part"] == "validation"

        if val_cfg == validation:
            if i_cfg in iterations:
                collisions[i_cfg] = cfg["predict_number"]

    if existing_predict_numbers:
        predict_number_new = max(existing_predict_numbers) + 1
    else:
        predict_number_new = 0
    for iteration in iterations:
        if collisions[iteration] is not None:
            predict_number = collisions[iteration]
        else:
            predict_number = predict_number_new
        print("Create {} for iteration {}".format("{}/setup_t{}_p{}".format(experiment, train_number, predict_number), iteration))

        set_up_environment(base_dir,
                           experiment,
                           train_number,
                           iteration,
                           predict_number,
                           clean_up,
                           validation)

        if collisions[iteration] is None:
            predict_number_new += 1

def set_up_environment(base_dir,
                       experiment,
                       train_number,
                       iteration,
                       predict_number,
                       clean_up,
                       validation):

    predict_setup_dir = os.path.join(os.path.join(base_dir, experiment), "03_predict/setup_t{}_p{}".format(train_number, predict_number))
    train_setup_dir = os.path.join(os.path.join(base_dir, experiment), "02_train/setup_t{}".format(train_number))
    train_checkpoint = os.path.join(train_setup_dir, "model_checkpoint_{}".format(iteration))
    train_config = os.path.join(train_setup_dir, "train_config.ini")
    train_worker_config = os.path.join(train_setup_dir, "worker_config.ini")

    if not os.path.exists(train_checkpoint):
        raise ValueError("No checkpoint at {}".format(train_checkpoint))

    if clean_up:
        if __name__ == "__main__":
            if click.confirm('Are you sure you want to remove {} and all its contents?'.format(predict_setup_dir), default=False):
                rmtree(predict_setup_dir)
            else:
                print("Abort clean up")

    else:
        if not os.path.exists(predict_setup_dir):
            os.makedirs(predict_setup_dir)
        else:
            if __name__ == "__main__":
                if click.confirm('Predict setup {} exists already, overwrite?'.format(predict_setup_dir), default=False):
                    rmtree(predict_setup_dir)
                    os.makedirs(predict_setup_dir)
                else:
                    print("Abort.")
                    return
            else:
                raise ValueError("Predict setup exists already, choose different predict number or clean up.")

        copyfile(os.path.join(this_dir, "synister/predict_pipeline.py"), os.path.join(predict_setup_dir, "predict_pipeline.py"))

        copyfile("synister/predict.py", os.path.join(predict_setup_dir, "predict.py"))
        copyfile(train_worker_config, os.path.join(predict_setup_dir, "worker_config.ini"))
     
        train_config_dict = read_train_config(train_config)

        predict_config = create_predict_config(base_dir,
                                               experiment,
                                               train_number,
                                               predict_number,
                                               train_checkpoint,
                                               train_config_dict,
                                               validation)

        with open(os.path.join(predict_setup_dir, "predict_config.ini"), "w+") as f:
            predict_config.write(f)

def create_predict_config(base_dir,
                          experiment,
                          train_number,
                          predict_number,
                          train_checkpoint,
                          train_config_dict,
                          validation):

    config = configparser.ConfigParser()

    config.add_section('Predict')
    config.set('Predict', 'train_checkpoint', train_checkpoint)
    config.set('Predict', 'experiment', str(experiment))
    config.set('Predict', 'train_number', str(train_number))
    config.set('Predict', 'predict_number', str(predict_number))

    synapse_types_string = ""
    for s in train_config_dict["synapse_types"]:
        synapse_types_string += s + ", "
    synapse_types_string = synapse_types_string[:-2]

    config.set('Predict', 'synapse_types', synapse_types_string)
    config.set('Predict', 'input_shape', '{}, {}, {}'.format(train_config_dict["input_shape"][0],
                                                             train_config_dict["input_shape"][1],
                                                             train_config_dict["input_shape"][2]))
    config.set('Predict', 'fmaps', str(train_config_dict["fmaps"]))
    config.set('Predict', 'batch_size', str(train_config_dict["batch_size"]))
    config.set('Predict', 'db_credentials', str(train_config_dict["db_credentials"]))
    config.set('Predict', 'db_name_data', str(train_config_dict["db_name_data"]))
    config.set('Predict', 'split_name', str(train_config_dict["split_name"]))
    config.set('Predict', 'voxel_size', "{}, {}, {}".format(train_config_dict["voxel_size"][0],
                                                            train_config_dict["voxel_size"][1],
                                                            train_config_dict["voxel_size"][2]))
    config.set('Predict', 'raw_container', str(train_config_dict["raw_container"]))
    config.set('Predict', 'raw_dataset', str(train_config_dict["raw_dataset"]))
    config.set('Predict', 'neither_class', str(train_config_dict["neither_class"]))
    config.set('Predict', 'downsample_factors', str(train_config_dict["downsample_factors"])[1:-1])
    if validation:
        config.set('Predict', 'split_part', "validation")
    else:
        config.set('Predict', 'split_part', "test")
    config.set('Predict', 'overwrite', str(False))
    config.set('Predict', 'network', train_config_dict["network"])
    config.set('Predict', 'fmap_inc', ", ".join(str(v) for v in train_config_dict["fmap_inc"]))
    config.set('Predict', 'n_convolutions', ", ".join(str(v) for v in train_config_dict["n_convolutions"]))
    config.set('Predict', 'network_appendix', train_config_dict["network_appendix"])
    
    return config
 
if __name__ == "__main__":
    options = p.parse_args()

    base_dir = options.base_dir
    experiment = options.e
    train_number = int(options.t)
    train_iterations = [int(i) for i in options.i]
    clean_up = bool(options.c)
    validation = bool(options.v)
    set_up_environments(base_dir,
                       experiment,
                       train_number,
                       train_iterations,
                       clean_up,
                       validation)
