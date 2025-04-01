"""Module to define CONSTANTS used across the DEEPaaS Interface.

This module is used to define CONSTANTS used across the API interface.
Do not misuse this module to define variables that are not CONSTANTS or
that are not used across the `api` package. You can use the `config.py`
file on your model package to define CONSTANTS related to your model.

By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""
import logging
import os
from pathlib import Path
from importlib import metadata
import yaml

# Necessary imports for api.__init__ and api.schemas code
# pylint: disable=unused-import
from thermal_anomaly_segmenter.config import DATA_PATH  # noqa: F401
from thermal_anomaly_segmenter.config import MODELS_PATH  # noqa: F401
from thermal_anomaly_segmenter.config import REMOTE_PATH  # noqa: F401
from thermal_anomaly_segmenter.config import MODEL_VARIANTS  # noqa: F401
from thermal_anomaly_segmenter.config import TRAIN_PHASES  # noqa: F401
from thermal_anomaly_segmenter.config import LOSS_FUNCTIONS  # noqa: F401
from thermal_anomaly_segmenter.config import MLFLOW_URI  # noqa: F401
from thermal_anomaly_segmenter.config import MLFLOW_USERNAME  # noqa: F401
from thermal_anomaly_segmenter.config import MLFLOW_PASSWORD  # noqa: F401


# Get AI model metadata from pyproject.toml
API_NAME = "thermal_anomaly_segmenter"
PACKAGE_METADATA = metadata.metadata(API_NAME)  # .json

# Get ai4-metadata.yaml metadata
CWD = os.getcwd()
try:
    AI4_METADATA_PATH = sorted(Path(CWD).rglob("ai4-metadata.yml"))[0]
    with open(AI4_METADATA_PATH, "r", encoding="utf-8") as stream:
        AI4_METADATA = yaml.safe_load(stream)
except IndexError:
    AI4_METADATA = {"description": "-"}

# Project metadata
PROJECT_METADATA = {
  "name": PACKAGE_METADATA["Name"],
  "description": AI4_METADATA["description"],
  "license": PACKAGE_METADATA["License"],
  "version":  PACKAGE_METADATA["Version"],
  "url":  PACKAGE_METADATA["Project-URL"],
}

# Fix metadata for authors and emails from pyproject parsing
_AUTHORMAILS_LIST = PACKAGE_METADATA["Author-email"].split(", ")
_AUTHORMAILS = dict(map(lambda s: s[:-1].split(" <"), _AUTHORMAILS_LIST))
try:
    _AUTHORS_LIST = PACKAGE_METADATA["Author"].split(", ")
    _AUTHORMAILS = {**_AUTHORMAILS, **{a: "" for a in _AUTHORS_LIST}}
except IndexError:
    pass
PROJECT_METADATA["author-email"] = _AUTHORMAILS
PROJECT_METADATA["author"] = ", ".join(_AUTHORMAILS.keys())

# logging level across API modules can be setup via API_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("API_LOG_LEVEL", default="INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())
