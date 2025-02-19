"""
Wrapper around neural-lam.train to allow passing parameters via yaml file, e.g. params.yaml
"""

import argparse
import os
import sys
import yaml

import dvc.api
from loguru import logger

from neural_lam.train_model import main as train_model


def main(params):
    train_model(params)

def none_or_str(value):
    if value == 'None':
        return None
    return value

# explicitly list the arguments that we want to use from the yaml file
# that will be passed to neural_lam.train_model.main()
# for any arguments that are commented out the default value in
# neural_lam.train_model.main() will be used
NEURAL_LAM_TRAIN_ARGS_TO_USE = [
    "config_path",
    "model",
    # "seed",
    "num_workers",
    "num_nodes",
    # "devices",
    "epochs",
    "batch_size",
    "load",
    # "restore_opt",
    # "precision",
    "graph",
    "hidden_dim",
    # "hidden_layers",
    # "processor_layers",
    # "mesh_aggr",
    # "output_std",
    "ar_steps_train",
    # "loss",
    "lr",
    # "val_interval",
    # "eval",
    "ar_steps_eval",
    # "n_example_pred",
    "logger",
    "logger_project",
    "val_steps_to_log",
    "metrics_watch",
    # "var_leads_metrics_watch",
    # "num_past_forcing_steps",
    # "num_future_forcing_steps",
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="params.yaml")
    parser.add_argument("--eval", type=none_or_str, default=None, choices=[None, "val", "test"])
    parser.add_argument("--load", type=none_or_str, default=None, help="Path to model checkpoint to load if evaluating")
    parser.add_argument("--logger", type=str, default="mlflow")
    parser.add_argument("--logger_project", type=str, default="neural_lam")
    options = parser.parse_args()
    if options.eval is not None:
        params = dvc.api.params_show(stages=["evaluate"])
    else:
        params = dvc.api.params_show(stages=["train"])
    params.update(vars(options))
    print(params)

    if options.eval is not None and options.load is None:
        raise ValueError("Must provide --load when evaluating")
    args = []
    for k in NEURAL_LAM_TRAIN_ARGS_TO_USE:
        v = params[k]
        if k == "logger_project":
            # in neural-lam this argument is called "project", not "logger_project"
            k = "logger-project"

        if isinstance(v, list):
            # need to turn these into multiple arguments, rather than a single
            # quoted string, otherwise argparse args that take a list of
            # arguments (like val_steps_to_log) will not work
            args.extend([f"--{k}", *map(str, v)])
        else:
            args.extend([f"--{k}", str(v)])
    # args = params
    logger.info(f"Initiating run with parameters: {args}")
    main(args)
