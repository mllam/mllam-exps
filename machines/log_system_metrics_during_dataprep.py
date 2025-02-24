import time
import psutil
import mlflow
import mllam_data_prep as mdp
import signal
import sys
import urllib3
import os
from pathlib import Path
from loguru import logger

# supress warnings about insecure requests to MLflow tracking server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

EXPERIMENT_NAME = "mllam-data-prep"

def _calc_directory_size(directory: str) -> int:
    """
    Calculate the size of a directory in bytes.

    Parameters
    ----------
    directory : str
        Path to the directory to calculate the size of.
    
    Returns
    -------
    int
        Size of the directory in bytes.
    """
    total = 0
    fp_root = Path(directory)
    if not fp_root.exists():
        return 0

    for entry in fp_root.rglob("*"):
        if entry.is_file():
            try:
                total += os.path.getsize(entry)
            except FileNotFoundError:
                # this can happen because dask writes temporary files that are deleted
                pass
    return total



def main(config_path: str, dataset_output_path: str):
    """
    Set up an MLflow experiment and log system metrics during data preparation.
    This doesn't actually do any "machine learning", but using mlflow to track
    system metrics is useful for debugging the dataset preperation.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file used with mllam_data_prep to generate
        the dataset. We log the configuration parameters to MLflow.
    dataset_output_path : str
        Path to the directory where the dataset is being generated. We log the
        size of this directory to MLflow as the dataset is being generated.
    """
    # Set experiment name
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load config
    config = mdp.Config.from_yaml_file(config_path)

    # Start MLflow Run
    run = mlflow.start_run(log_system_metrics=True)
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    
    logger.info(f"Logging system metrics to MLflow experiment '{EXPERIMENT_NAME}'")
    logger.info(f"Run id: {run.info.run_id}, experiment id: {experiment_id}")

    mlflow.log_params(config.to_dict())
    
    mlflow.enable_system_metrics_logging()

    def handle_exit(signum, frame):
        """Gracefully handle script termination."""
        print("\nReceived termination signal. Closing MLflow run...")
        mlflow.end_run()
        sys.exit(0)

    # Register the signal handler for SIGINT (CTRL+C) and SIGTERM (kill command)
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    

    try:
        step = 0  # Initialize step counter
        while True:
            # compute the size of the dataset and log it
            dataset_size = _calc_directory_size(dataset_output_path)
            mlflow.log_metric("dataset_size_bytes", dataset_size, step=step)
            step += 1  # Manually increment step counter

            time.sleep(10)  # Log every 10 seconds

    except KeyboardInterrupt:
        handle_exit(None, None)  # Ensures graceful shutdown if interrupted manually


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, required=True)
    argparser.add_argument("--dataset_output_path", type=str, required=True)
    args = argparser.parse_args()
    
    main(config_path=args.config_path, dataset_output_path=args.dataset_output_path)