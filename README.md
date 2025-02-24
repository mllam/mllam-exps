# Orchestrating MLLAM Experiments with DVC

This repository contains a [DVC](dvc.org) pipeline to train, reproduce and share experiments for the Neural-LAM project.

>[!Warning]
>This package is still under heavy development and currently has also a tendency to be geared towards a specific HPC environment.

## Design
Pipelines consist of several stages, each stage describes a single step in the experiment process. Main stages for an ML experiment consist
typically of data preparation, model training, evaluation and inference. These stages are defined in the `dvc.yaml` file. For the Neural-LAM
project, the main stages are _prepare dataset_, _create graph_, _train_, _evaluate_. The stages are linked by dependencies (`deps`) and outputs (`out`).
DVC automatically tracks the dependencies and outputs and only runs stages that have been updated. In the background DVC uses git for these purposes.
In addition, also paramters like model version, num_workers, epochs, ... need to be tracked. These are defined in `params.yaml` files, here
particularly in `data/training_params.yaml` and `data/evaluate_params.yaml`.

Generally, experiments can be run with `dvc exp run`, but since we need to schedule our experiments with SLURM and DVC relies on being called
after a job has been finished to calculate the checksums, we need to do some hacks:

1. Use `sbatch -W` in the `cmd` field of the stage to return to DVC only after the model tasks has been finished.
2. Use a tmux session to run DVC in the background and let it wait for the job to finish.
3. Use wrapper scripts to ingest SLURM environment variables and setups.

The overall structure of the repository is as follows:

```plaintext
mllam-exps
├── data
│   ├── datastore.zarr  # output from the prepare dataset stage (not there initially)
│   ├── graph  # output from the create graph stage (not there initially)
│   ├── config.yaml  # configuration file for the Neural-LAM project
│   └── datastore.yaml  # configuration file for the data preparation stage via mllam-data-prep
├── logs  # log files from SLURM jobs
├── machines  # machine configurations for the Neural-LAM project
│   ├── environment.sh  # source environment incl. python modules
│   ├── check_for_venv_path.sh  # script to check if path to virtual python environment is set 
│   ├── slurm.neural-lam.sh  # SLURM wrapper script for training and evaluation
│   └── slurm.mllam-data-prep.sh  # SLURM wrapper script for mllam-data-prep
├── dvc.lock  # lock file for DVC containing checksums for the latest stages
├── dvc.yaml  # DVC pipeline definition
├── params.yaml  # Parameters to train and evaluate model, which are converted to command line arguments to neural_lam
├── version.mllam.txt  # Lock file for neural-lam version (external dependencies are not well supported by DVC)
```

## Setup
To setup DVC, DVC needs to be installed:
```bash
pip install dvc
```

To be able to track experiments with MLflow, add your username and password to a `~/.bashrc`:
```
#!/bin/bash
export MLFLOW_TRACKING_USERNAME=""
export MLFLOW_TRACKING_PASSWORD=""
#empty line
```

Install neural-lam and dependencies into a virtual environment, and set the
environment variable `MLLAM_VENV_PATH` to the path of the virtual environment, e.g.:
```bash
export MLLAM_VENV_PATH=/dcai/users/denlef/git-repos/mllam/mllam-exps/.venv
```
A good place for this is also the `~/.bashrc`

### Sharing a DVC cache
This repository is set up for a common DVC cache, which allows to automatically pull the results of stages and experiments
someone else has already run. E.g. in the case of the baseline dataset, only the first person had to create this dataset
for all other team members this dataset will be pulled automatically from cache.

For the cache to work properly, ensure that you are using a group that all members are using, so that there won't be
permission errors. A good way to ensure this, add the following to the `~/.bashrc`:

```bash
# `newgrp` launches a new shell, so to avoid an infinite loop, check if the group is already set
if [ "$(id -gn)" != "cu_0003" ]; then
    newgrp cu_0003
fi
```
where `cu_0003` is the group that all team members belong to.

### Sharing experiments and data
To share experiments including models and metrics, a common remote needs to be defined. DVC supports a long list of
remotes, but on a cluster system that potentially does not have access to the internet and e.g. S3 buckets,
a local remote can be used. To setup a local remote, run:

```bash
dvc remote add -d localshare /dcai/projects/cu_0003/dvc/
```

When the remote is created for the first time (i.e. by the first user creating this folder), it needs to be initialized as git repository
and the access rights configured for group write and read:

```bash
cd /dcai/projects/cu_0003/dvc/
chmod o+rwx .
git init
```

Further info can be found in the [DVC documentation](https://dvc.org/doc/user-guide/experiment-management/sharing-experiments).

## Running an experiment

Experiments are generally run by:

```bash
dvc exp run
```

> **Note**: It is highly advised to run this in a tmux session as the command will take a while to finish (`sbatch -W`).

DVC automatically runs only the stages that have been updated as it keeps track of the dependencies defined in dvc.yaml.

Once an experiment has been run and it is worth sharing with others, it can be pushed to the remote:

```bash
dvc exp push /dcai/projects/cu_0003/dvc/ <experiment>
```
This command will push all tracked experiment data (model, graph, data, ...) to the remote.

In the same way, experiments can be pulled from the remote:

```bash
dvc exp pull /dcai/projects/cu_0003/dvc/ <experiment>
```

And if one is just curious which experiments are available:

```bash
dvc exp list /dcai/projects/cu_0003/dvc/ # on the remote
dvc exp list # in the user/local space
```

### Running a range of experiments
For hypterparameter searches or testing different configurations, it is possible to run a range of experiments. This can be done
by using the `--queue` option:

```bash
dvc exp run --queue -S data/training_params.yaml:hidden_dim='1,2,4,8,16'
dvc queue start
```

This will run the training stage with different hidden dimensions. The `--queue` option will create a sequence of experiments that
are defined by changing the parameters on-the-fly defined in e.g. `data/training_params.yaml` with the `-S` option.

>[!Info]
>If a stage is failing and has been run via `--temp`, it is [currently not saved and lost](https://github.com/iterative/dvc/issues/10616).
>Run your experiment therefore in the queue or workspace.


### Restarting from a checkpoint
To continue training on a previous experiment a few steps are necessary with the current setup:
1. Checkout the experiment that shall be continued with `dvc exp apply <experiment_name/hash>`
2. Persist the checkpoints in `saved_models` by adding the `persist: true` flag to the `train` and `evaluate` stage if not yet added.
    This ensures that the checkpoints are not cleared between experiments as this would normally be done.
3. Add `--load ./saved_models/*/last.ckpt --restore_opt` to the `train.cmd` to let neural-lam start from the checkpoint
4. Patch neural_lam to read run_name from previous run, e.g. with https://github.com/mllam/neural-lam/commit/4927a5f73172a41a4f009ab3be6d36ffbec8f6dd
    This ensures that the checkpoints are updated (dvc keeps tracks of the previous one if still needed) and we are never have several
    subfolders in `saved_models` making `saved_models/*/last.ckpt` always unique.
5. Increase the number of epochs in `params.yaml`. Remember that the number of epochs are absolute and not just relative to the last checkpoint.

>[!Info]
>This stage can currently only be run in the workspace and not via `--temp` or in the `queue` [Discord-Conversation](https://discord.com/channels/485586884165107732/563406153334128681/1343195392908726272)
