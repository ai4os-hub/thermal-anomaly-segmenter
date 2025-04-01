"""Package to create datasets, pipelines and other utilities.

You can use this module for example, to write all the functions needed to
operate the methods defined at `__init__.py` or in your scripts.

All functions here are optional and you can add or remove them as you need.
"""
from copy import deepcopy
import csv
import json
import logging
from pathlib import Path
import subprocess  # nosec B404
from tqdm import tqdm
import zipfile

from thermal_anomaly_segmenter import config as cfg

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


class ShortNameFormatter(logging.Formatter):
    def format(self, record):
        # Extract only the last part of the module name
        record.shortname = record.name.split('.')[-1]
        return super().format(record)


def configure_logging(logger, log_level: int):
    """Define basic logging configuration

    :param logger: logger
    :param log_level: User defined input
    """
    if logger.hasHandlers():
        return  # Prevent adding duplicate handlers

    log_format = ShortNameFormatter(
        '%(asctime)s - %(shortname)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Define logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)


def get_freest_gpu():
    """Returns the GPU ID with the most available memory."""
    try:
        # Run nvidia-smi and parse memory usage
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True
        )
        free_memory = [int(x) for x in result.stdout.strip().split("\n")]

        if not free_memory:
            return None  # No GPUs detected
        # Return GPU with max free memory
        return free_memory.index(max(free_memory))

    except Exception as e:
        print(f"Could not determine free GPU: {e}")
        return None  # Fail gracefully


def get_filestem(filepath: Path):
    """Get name of file without any suffixes"""
    return str(filepath.name).removesuffix(''.join(filepath.suffixes))


def check_path_exists(path: Path):
    """Raise error if path does not exist"""
    if not path.exists():
        raise FileNotFoundError(f"Path '{path}' does not exist!")


def load_config(config_path: str):
    """
    Load the config from the given path

    :param config_path: path to the config file
    :return: dictionary with the config
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def update_config(config: dict, **kwargs) -> dict:
    """
    Update the given JSON config dictionary with values from kwargs.
    Only specified keys are updated, others remain unchanged.

    :param config: Original JSON configuration as a dictionary.
    :param kwargs: Key-value pairs to update in the config.
    :return: Updated config dictionary.
    """
    updated_config = deepcopy(config)

    # Reverse mapping: JSON config keys to marshmallow field values
    field_mapping = {
        "config_name": (
            cfg.MODEL_VARIANTS[kwargs["model_type"]] + kwargs["loss_fn"]
        ),
        "output_path": str(cfg.MODELS_PATH),
        "datasets.main_dataset.augment_train": kwargs["data_augmentation"],
        "datasets.manual_dataset.augment_train": kwargs["data_augmentation"],
        "loss_fn": cfg.LOSS_FUNCTIONS[kwargs["loss_fn"]],
        "batch_size": kwargs["batch_size"],
        "worker_count": kwargs["worker_count"],
        "datasets.main_dataset.path": (
            f"{kwargs['dataset_path']}/generated_set"
        ),
        "datasets.manual_dataset.path": (
            f"{kwargs['dataset_path']}/manual_set"
        ),
    }

    tp_config = config["train_phases"]
    updated_tp_config = {
        k: v for k, v in tp_config.items() if k in kwargs["train_phases"]
    }
    field_mapping = {
        **field_mapping,
        **{"train_phases": updated_tp_config},
        **{"train_phase_order": kwargs["train_phases"]}
    }

    def set_nested_value(d, key_path, value):
        """Sets a value in a nested dictionary given a dot-separated path."""
        keys = key_path.split(".")
        for key in keys[:-1]:
            d = d.setdefault(key, {})  # Traverse / create nested dictionaries
        d[keys[-1]] = value

    for config_key, value in field_mapping.items():
        set_nested_value(updated_config, config_key, value)

    return updated_config


def unzip(zip_paths: list):
    """
    Unzip files to their current directory and delete
    the .zip afterwards

    Args:
        zip_paths (list): .zip files to extract
    """

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            logger.info(f"Unzipping '{zip_path}'")

            # Extract all files in .zip with progress bar
            files_in_zip = zip_ref.infolist()
            file_amount = len(files_in_zip)

            with tqdm(
                total=file_amount, unit='file',
                desc=f"Unzipping {zip_path.name}"
            ) as pbar:
                for f in files_in_zip:
                    try:
                        zip_ref.extract(f, zip_path.parent)
                        pbar.update(1)
                    except zipfile.error as e:
                        logger.error(f"Error extracting {f.filename}: {e}")

        # Remove .zip file after extraction
        logger.info(f"Extraction complete. Removing '{zip_path}'...")
        zip_path.unlink()


def extract_log_dict(csv_log_path: str):
    """Extract logged metrics from metrics.csv file"""

    with open(csv_log_path, mode='r') as file:
        reader = csv.DictReader(file)
        metrics_dict = [row for row in reader]

    results_dict = {
        k: v for k, v in metrics_dict[-1].items()
        if "test" in k
    }
    return results_dict
