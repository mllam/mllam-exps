# This script checks if the MLLAM_VENV_PATH is set and if not, it exits with an error message.

if [ -z "$MLLAM_VENV_PATH" ]; then
  echo "Please set the MLLAM_VENV_PATH environment variable to path of the virtual environment where you have neural-lam etc installed"
  exit 1
else
  echo "Using venv in $MLLAM_VENV_PATH"
fi

export DVC_WORKING_DIR=$(pwd)