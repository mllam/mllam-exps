#!/bin/bash -l
#SBATCH --job-name=HAS-NeuralLam
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8  #per node
#SBATCH --no-requeue
#SBATCH --partition=prodq
#SBATCH --exclusive
#SBATCH --account=cu_0003
#SBATCH --output=/dcai/users/schhau/git-repos/mllam-exps/logs/neurallam.%j.log
#SBATCH --error=/dcai/users/schhau/git-repos/mllam-exps/logs/neurallam.%j.log

echo "Started slurm job $SLURM_JOB_ID"

<<<<<<< HEAD
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

srun -ul python train_wrapper.py "$@"
