#!/bin/bash -l
#SBATCH --job-name=HAS-NeuralLam
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8  #per node
#SBATCH --no-requeue
#SBATCH --partition=defq
#SBATCH --exclusive
#SBATCH --account=cu_0003
#SBATCH --output=/dcai/users/%u/logs/neurallam.%j.log
#SBATCH --error=/dcai/users/%u/logs/neurallam.%j.log

echo "Started slurm job $SLURM_JOB_ID"

export CARTOPY_DATA_DIR=/dcai/projects/cu_0003/user_space/has/cartopy_features/
export MLFLOW_TRACKING_URI="https://mlflow.dmi.dcs.dcai.dk" #sqlite:///mlflow.db #
export MLFLOW_TRACKING_INSECURE_TLS=true

source machines/secrets.sh
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

# remove the --venv_path argument from the args passed to this script and
# assign it to the VENV_PATH variable
args=("$@")
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--venv_path" ]]; then
        VENV_PATH="${args[$i+1]}"
        unset 'args[i]'
        unset 'args[i+1]'
    fi
done

echo "Using venv in ${VENV_PATH}"

# source the virtual environment so that the python script can be run
source ${VENV_PATH}/bin/activate

# pass the rest of the args to the python script
srun -ul python train_wrapper.py "${args[@]}"
