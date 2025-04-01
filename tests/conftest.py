"""Generic tests environment configuration. This file implement all generic
fixtures to simplify model and api specific testing.

Modify this file only if you need to add new fixtures or modify the existing
related to the environment and generic tests.
"""
# pylint: disable=redefined-outer-name
import inspect
import os
import shutil
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from unittest.mock import patch
import pytest

import api

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session", autouse=True)
def original_datapath():
    """Fixture to generate a original directory path for datasets."""
    return Path(api.config.DATA_PATH).absolute()


@pytest.fixture(scope="session", autouse=True)
def original_modelspath():
    """Fixture to generate a original directory path for datasets."""
    return Path(api.config.MODELS_PATH).absolute()


@pytest.fixture(scope="session", params=os.listdir("tests/configurations"))
def config_file(request):
    """Fixture to provide each deepaas configuration path."""
    config_str = f"tests/configurations/{request.param}"
    return Path(config_str).absolute()


@pytest.fixture(scope="module", name="testdir")
def create_testdir():
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as testdir:
        os.chdir(testdir)
        yield testdir


@pytest.fixture(scope="module", autouse=True)
def copytree_data(testdir, original_datapath):
    """Fixture to copy the original data directory to the test directory."""
    shutil.copytree(original_datapath, f"{testdir}/{api.config.DATA_PATH}")


@pytest.fixture(scope="module", autouse=True)
def copytree_models(testdir, original_modelspath):
    """Fixture to copy the original models directory to the test directory."""
    shutil.copytree(original_modelspath, f"{testdir}/{api.config.MODELS_PATH}")


@pytest.fixture(scope="module", autouse=True)
def copytree_configs(testdir, original_modelspath):
    """Fixture to copy the original models directory to the test directory."""
    cfgs_subpath = Path("thermal_anomaly_segmenter", "configurations")
    original_models_cfgspath = Path(original_modelspath.parents[1],
                                    cfgs_subpath)
    shutil.copytree(
        original_models_cfgspath,
        f"{testdir}/{api.config.MODELS_PATH.parent}/{cfgs_subpath}"
    )


def generate_signature(names, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD):
    """Function to generate dynamically signatures."""
    parameters = [inspect.Parameter(name, kind) for name in names]
    return inspect.Signature(parameters=parameters)


def generate_fields_fixture(signature):
    """Function to generate dynamically fixtures with dynamic arguments."""
    def fixture_function(**options):  # fmt: skip
        return {k: v for k, v in options.items() if v is not None}
    fixture_function.__signature__ = signature
    return pytest.fixture(scope="module")(fixture_function)


# PATCHING - GENERAL
@pytest.fixture(scope="module")
def patch_get_dirs():
    """Patch to replace get_dirs"""

    with patch(
        "api.utils.get_dirs", autospec=True
    ) as mock_get_dirs:
        mock_get_dirs.return_value = ["dummy/folder1/"]
        yield mock_get_dirs


# PATCHING - FOR POST "PREDICT"
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=1,
            kernel_size=3, padding=1
        )
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.conv(x)
        self.logits = x
        return self


@pytest.fixture(scope="module")
def patch_load_model():
    """Patch to replace load_model"""
    with patch(
        "thermal_anomaly_segmenter.prediction.load_model",
        autospec=True
    ) as mock_load_model:
        mock_load_model.return_value = (
            DummyModel().to(DEVICE).eval()
        )
        yield mock_load_model


# PATCHING - FOR POST "TRAIN"
@pytest.fixture(scope="module")
def patch_train_segmentation(testdir):
    """Patch to replace patch_train_segmentation (train execution)"""

    with patch(
        "thermal_anomaly_segmenter.train_segmentation", autospec=True
    ) as mock_train_segmentation:
        # create dummy training folder and content
        mock_model_path = Path(
            testdir,
            api.config.MODELS_PATH,
            f"segfB0PolyDice_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        mock_model_path.mkdir(exist_ok=True)

        with open(
            Path(mock_model_path, "metrics.csv"), "w"
        ) as mock_csv_file:
            mock_csv_file.write("epoch,metric\n0,0")

        # run mock training
        mock_train_segmentation.return_value = mock_model_path
        yield mock_train_segmentation


# #############################################
# RUN THE TESTS FOR get_metadata, predict AND train
@pytest.fixture(scope="module")
def metadata(patch_get_dirs):
    """Fixture to return get_metadata to assert properties."""
    return api.get_metadata()


# Generate and inject fixtures for predict arguments
fields_predict = api.schemas.PredArgsSchema().fields
signature = generate_signature(fields_predict.keys())
globals()["predict_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def predictions(patch_get_dirs, patch_load_model, predict_kwds):
    """Fixture to return predictions to assert properties."""
    print(predict_kwds["input_file"])
    return api.predict(**predict_kwds)


# Generate and inject fixtures for training arguments
fields_training = api.schemas.TrainArgsSchema().fields
signature = generate_signature(fields_training.keys())
globals()["training_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def training(patch_train_segmentation, training_kwds):
    """Fixture to return training to assert properties."""
    return api.train(**training_kwds)
