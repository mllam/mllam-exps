import time
import psutil
import mlflow
import mllam_data_prep as mdp
import signal
import sys
import urllib3
import numpy as np
import sklearn.linear_model  # Dummy model to trigger autologging

# supress warnings about insecure requests to MLflow tracking server
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

EXPERIMENT_NAME = "mllam-data-prep"

def main(config_path):
    # Set experiment name
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load config
    config = mdp.Config.from_yaml_file(config_path)

    # Start MLflow Run
    mlflow.start_run()

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
    
    # Dummy model training to activate system monitoring
    X_dummy = np.random.rand(100, 5)
    y_dummy = np.random.rand(100)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_dummy, y_dummy)  # This triggers MLflow's system monitoring

    try:
        step = 0  # Initialize step counter
        while True:
            # Collect system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            ram_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage("/").percent

            # Log system metrics with step count
            mlflow.log_metric("CPU_Usage", cpu_usage, step=step)
            mlflow.log_metric("RAM_Usage", ram_usage, step=step)
            mlflow.log_metric("Disk_Usage", disk_usage, step=step)

            step += 1  # Manually increment step counter

            time.sleep(2)  # Log every 10 seconds

    except KeyboardInterrupt:
        handle_exit(None, None)  # Ensures graceful shutdown if interrupted manually


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config_path", type=str, required=True)
    args = argparser.parse_args()
    
    main(config_path=args.config_path)