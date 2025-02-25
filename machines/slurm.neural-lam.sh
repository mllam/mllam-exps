#!/bin/bash -l
#SBATCH --output=/dcai/users/%u/logs/neurallam.%j.log
#SBATCH --error=/dcai/users/%u/logs/neurallam.%j.log

# chdir in the slurm directive is to make sure the gpu stats file is saved there
# Then we cd to the actual working directory

cd ${DVC_WORKING_DIR}

echo "Started slurm job $SLURM_JOB_ID"

export CARTOPY_DATA_DIR=/dcai/projects01/cu_0003/data/cartopy_features
export MLFLOW_TRACKING_URI="https://mlflow.dmi.dcs.dcai.dk" #sqlite:///mlflow.db #
export MLFLOW_TRACKING_INSECURE_TLS=true

source machines/environment.sh

set -a
LOGLEVEL=INFO
CUDA_LAUNCH_BLOCKING=1

OMPI_MCA_pml=ucx
OMPI_MCA_btl=^vader,tcp,openib,uct
UCX_NET_DEVICES=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
NCCL_SOCKET_IFNAME=ens6f0
NCCL_IB_HCA=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11
OMP_NUM_THREADS=56
OMPI_MCA_coll_hcoll_enable=0
set +a

echo "Using venv in ${MLLAM_VENV_PATH}"

# source the virtual environment so that the python script can be run
source ${MLLAM_VENV_PATH}/bin/activate

# Check if 'eval' is in the arguments
if [[ " $@ " == *"--eval"* ]]; then
        MODE="eval"
    else
        MODE="train"
fi

# pass all arguments to the python script
srun -ul -K1 python -m neural_lam.train_model --logger_run_name $MODE-$DVC_EXP_NAME "$@"
