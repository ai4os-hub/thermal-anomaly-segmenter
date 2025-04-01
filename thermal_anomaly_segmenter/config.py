"""Module to define CONSTANTS used across the AI-model package.

This module is used to define CONSTANTS used across the AI-model package.
Do not misuse this module to define variables that are not CONSTANTS or
exclusive to the thermal_anomaly_segmenter package. You can use the `config.py`
inside `api` to define exclusive CONSTANTS related to your interface.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""
# Do NOT import anything from `api` or `thermal_anomaly_segmenter` here.
# That might create circular dependencies.
import logging
import os
from pathlib import Path

# DEEPaaS can load more than one installed models. Therefore, in order to
# avoid conflicts, each default PATH environment variables should lead to
# a different folder. The current practice is to use the path from where the
# model source is located.
BASE_PATH = Path(__file__).resolve(strict=True).parents[1]

# Path definition for the pre-trained models
MODELS_PATH = os.getenv("MODELS_PATH", default=BASE_PATH / "models")
MODELS_PATH = Path(MODELS_PATH)
# Path definition for data folder
DATA_PATH = os.getenv("DATA_PATH", default=BASE_PATH / "data")
DATA_PATH = Path(DATA_PATH)

# Remote (rshare) paths for data and models
REMOTE_PATH = os.getenv("REMOTE_PATH", default="/storage/taseg")
REMOTE_DATA_PATH = os.getenv("REMOTE_DATA_PATH",
                             default=os.path.join(REMOTE_PATH, "data"))
REMOTE_DATA_PATH = Path(REMOTE_DATA_PATH)
REMOTE_MODELS_PATH = os.getenv("REMOTE_MODELS_PATH",
                               default=os.path.join(REMOTE_PATH, "models"))
REMOTE_MODELS_PATH = Path(REMOTE_MODELS_PATH)

# configure logging:
# logging level across various modules can be setup via USER_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())

# MLFlow parameters
MLFLOW_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI",
                       "https://mlflow.cloud.ai4eosc.eu/")
MLFLOW_EXPERIMENT_NAME = BASE_PATH.name

# Model variants
MODEL_VARIANTS = {
    "segformer_b0": "segfB0Poly",
    "segformer_b1": "segfB1Poly",
    "segformer_b2": "segfB2Poly",
    "segformer_b3": "segfB3Poly",
    "segformer_b4": "segfB4Poly",
    "smp_deeplabV3plus": "dv3+Poly",
    "smp_unet": "unetPoly",
    "smp_pspnet": "pspnetPoly"
}
# Train phases
TRAIN_PHASES = [
    "pre_train_frozen",
    "pre_train_unfrozen",
    "fine_tune"
]
# Loss function variants
LOSS_FUNCTIONS = {
    "Tversky": "tversky_0.3_0.7",
    "Dice": "dice",
    "Jaccard": "jaccard",
    "BCE": "bce"
}
