#!/bin/bash -l
#SBATCH --job-name=mllam-data-prep
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account=cu_0003
#SBATCH --output=/dcai/users/%u/logs/neurallam.%j.log
#SBATCH --error=/dcai/users/%u/logs/neurallam.%j.log

echo "Started slurm job $SLURM_JOB_ID"

source machines/environment.sh

# Export for stability
export OMPI_MCA_coll_hcoll_enable=0

echo "Using venv in ${MLLAM_VENV_PATH}"

# source the virtual environment so that the python script can be run
source ${MLLAM_VENV_PATH}/bin/activate

# the config path is the first argument for mllam_data_prep
CONFIG_PATH=$1

python machines/log_system_metrics_during_dataprep.py --config_path=${CONFIG_PATH} & LOGGER_PID=$!
srun -ul python -m mllam_data_prep "$@"
kill $LOGGER_PID