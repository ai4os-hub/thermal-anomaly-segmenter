"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions. You can
use and edit any of the defined functions to improve or add methods to
your API.

The module shows simple but efficient example utilities. However, you may
need to modify them for your needs.
"""
import itertools
import logging
import os
import psutil
import requests
import sys

from . import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


class MLFlowAuthenticationError(Exception):
    """Custom exception for authentication failure"""
    pass


def ls_dirs(path):
    """Utility to return a list of directories available in `path` folder.

    Arguments:
        path -- Directory path to scan for folders.

    Returns:
        A list of strings for found subdirectories.
    """
    logger.debug("Scanning directories at: %s", path)
    dirscan = (x.name for x in path.iterdir() if x.is_dir())
    return sorted(dirscan)


def ls_files(path, pattern):
    """Utility to return a list of files available in `path` folder.

    Arguments:
        path -- Directory path to scan.
        pattern -- File pattern to filter found files. See glob.glob() python.

    Returns:
        A list of strings for files found according to the pattern.
    """
    logger.debug("Scanning for %s files at: %s", pattern, path)
    dirscan = (x.name for x in path.glob(pattern))
    return sorted(dirscan)


def get_dirs(root_dir: str, patterns: set = {}):
    """Utility to return a list of directories containing
    specific folder / file entries.
        - get_dirs(root_dir=config.DATA_PATH,
                   patterns={'images', 'annotations'})
        - get_dirs(root_dir=config.REMOTE_PATH,
                   patterns={'UNet.hdf5'})

    Arguments:
        root_dir (str): directory path to scan
        patterns (set): entry patterns to search for, defaults to {}
    """
    dirscan = [
        root for root, dirs, files in os.walk(root_dir)
        if patterns <= set(dirs) or patterns <= set(files)
    ]
    return sorted(dirscan)


def get_dirs_(root_dir: str, list_of_patterns: list):
    """Run get_dirs with a list of patterns and concatenate results."""
    total_dirscan = [
        get_dirs(root_dir, p) for p in list_of_patterns if type(p) is set
    ]
    return list(itertools.chain(*total_dirscan))


def get_default_path(directory_list: list, patterns):
    """
    Utility to get the default path for marshmallow fields

    Args:
        directory_list (list): Path directories to be checked
        patterns: folders / files to look for

    Returns:
        A default path for the marshmallow field in question
    """
    try:
        for d in directory_list:

            if isinstance(patterns, set):
                default_paths = get_dirs(d, patterns=patterns)
            elif isinstance(patterns, list):
                default_paths = get_dirs_(d, list_of_patterns=patterns)
            else:
                raise ValueError(
                    f"'patterns' is a '{type(patterns)}', not set or list!"
                )

            if default_paths != []:
                return default_paths[0]

    except IndexError:
        return None


def check_mlflow_authentication():
    """Make sure MLFlow connection works by sending a simple request
    """

    try:
        response = requests.get(
            f'{config.MLFLOW_URI}',
            auth=(config.MLFLOW_USERNAME, config.MLFLOW_PASSWORD),
            timeout=60
        )
        if response.status_code == 200:
            logger.info("MLFlow authentication successful!")
            return True
        else:
            raise MLFlowAuthenticationError(
                f"Authentication failed: {response.status_code} - "
                f"{response.text}"
            )

    except requests.exceptions.RequestException as e:
        raise MLFlowAuthenticationError(
            f"Error connecting to MLflow server: {e}"
        )


def get_optimal_num_workers():
    """Get optimal number of workers based on deployment setup"""
    num_cpu_cores = os.cpu_count()
    max_workers_based_on_cpus = num_cpu_cores - 2

    total_ram = psutil.virtual_memory().total
    min_ram_per_worker = 2 * 1024**3    # 2 GB per worker
    max_workers_based_on_ram = total_ram // min_ram_per_worker

    num_workers = min(max_workers_based_on_cpus,
                      max_workers_based_on_ram)
    return int(num_workers // 4)    # reduce to prevent overtaxation


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        logger.debug("Web args schema: %s", schema)
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema
