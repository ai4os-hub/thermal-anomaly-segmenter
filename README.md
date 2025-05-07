# thermal-anomaly-segmenter

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/thermal-anomaly-segmenter/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/thermal-anomaly-segmenter/job/main)

This repository can be used to infer with or train a semantic segmentation model,
specifically a [SegFormer](https://huggingface.co/docs/transformers/)
or [SMP model](https://github.com/qubvel-org/segmentation_models.pytorch) such as the DeepLabV3+.
It's purpose is the detection of thermal anomalies that may pertain to leaks in district
heating systems (DHSs).
The models are therefore trained on thermal imagery acquired by unmanned aircraft system (UAS),
which is processed to consist of three channels: (T_m, T_m, T_u). T_m is the thermal image
masked with the DHS pipeline location and T_u is the full, unmasked image.

https://github.com/user-attachments/assets/9bb405bf-fcbc-4ea1-9bec-910595f95c30

The repo is also the DEEPaaS API for the ["Thermal Anomaly Segmenter (TASeg)" module on the AI4EOSC platform](https://dashboard.cloud.ai4eosc.eu/catalog/modules/thermal-anomaly-segmenter),
described as "UAS-based thermal urban anomaly semantic segmentation for leak detection in DHSs".

## Installation

The simplest way to use the module is via the AI4EOSC platform or locally with the
[Docker image](https://hub.docker.com/r/ai4oshub/thermal-anomaly-segmenter) (which is
also buildable via the repo's Dockerfile).

Alternatively, the bash script `setting_up_deployment.sh` can be run to install everything
automatically in an AI4EOSC development deployment:
```bash
wget https://raw.githubusercontent.com/ai4os-hub/thermal-anomaly-segmenter/main/setting_up_deployment.sh
source setting_up_deployment.sh
```
This takes care of all required installations and finishes by running
[deepaas](https://github.com/ai4os/DEEPaaS).

## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── Dockerfile              <- Steps to build a DEEPaaS API Docker image
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── thermal_anomaly_segmenter
│   ├── README.md           <- Description of the TASeg.
│   ├── __init__.py         <- Makes thermal_anomaly_segmenter a Python module
│   ├── ...                 <- Other source code files
│   └── config.py           <- Module to define CONSTANTS used across the AI-model python package
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data subpackage for the integration with DEEP API
│
├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                  <- Folder to store your models (alternatively, store remotely)
│
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
│                              the creator's initials (if many user development),
│                              and a short `_` delimited description, e.g.
│                              `1.0-jqp-initial_data_exploration.ipynb`.
│
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to used by tox for running code tests
├── requirements.txt        <- Requirements file to run the API and models
│
├── pyproject.toml          <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── models              <- Folder to store the models for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│   └── test_training       <- Tests folder for model training requirements
│
└── tox.ini                 <- tox file with settings for running tox; see tox.testrun.org
```

## Usage

### Setting environment variables

If you wish to deviate from the general, given setup, you 
can view and change the defined environment variables in `./thermal_anomaly_segmenter/config.py`.
These include:

- `DATA_PATH`: Path to data folder. *Default: `./data`.*
- `MODELS_PATH`: Path to folder for saving trained models. *Default: `./models`.*
- `REMOTE_PATH`: Path to folder on remote directory containing `models` and / or `data`. *Default: `/storage/taseg`.*
- `MODEL_VARIANTS`: Default model variant options for training. *Default: SegFormer, DeepLabV3+.*
- `LOSS_FUNCTIONS`: Define loss function options for training. *Default: Tversky, Dice, BCE, Jaccard.*

In addition, the following variables are defined for MLFlow experiment logging and tracking:

- `MLFLOW_TRACKING_URI`: URL to MLFlow server. *Default: os.getenv or `https://mlflow.cloud.ai4eosc.eu/`.*
- `MLFLOW_TRACKING_USERNAME`: Username for login. *Default: os.getenv.*
- `MLFLOW_TRACKING_PASSWORD`: Password for login. *Default: os.getenv.*

The MLFlow-related environment variables are automatically set according to your
[MLFlow credentials](https://mlflow.cloud.ai4eosc.eu/signup).
You can check their definitions via:
```bash
$ printenv | grep -i MLFLOW
```
If they haven't been defined properly, you can set them for your current terminal via:
```bash
$ export $MLFLOW_TRACKING_USERNAME="your username"
$ export $MLFLOW_TRACKING_PASSWORD="your password"
```

### Via the AI4EOSC platform and DEEPaaS

The folder `thermal_anomaly_segmenter` contains the model's source code and all
relevant modules, methods and functions for training and inference.
For more information, see the [README.md](./thermal_anomaly_segmenter/README.md).

These methods are used by the subpackage `api` to define the API interface.
The API and CLI arguments are customized via the `api.schemas` and `api.responses`
modules.

Running `deep-start` or `nohup deep-start &` will allow access to the Swagger UI
to use the different functionalities.

### Directly via CLI

While this repository was designed for modular usage with DEEPaaS, you can also run
inference and prediction directly as scripts via command line.
The `cli.py` enables this functionality. Two commands exist, `train` and `predict`,
that can be selected for direct execution:

```bash
$ python -m cli {train,predict} {arguments}
```

For specific information on the required arguments and their default values, run:
```bash
$ python -m cli train --help
$ python -m cli predict --help
```

## Testing

Testing process is automated by tox library. You can check the environments
configured to be tested by running `tox --listenvs`. Tests are implemented
following [pytest](https://docs.pytest.org) framework.
Fixtures and parametrization are placed inside `conftest.py` files, while
assertion tests are located on `test_*.py` files.

Running the tests with tox:

```bash
$ pip install -r requirements-dev.txt
$ tox
```

Running the tests with pytest:

```bash
$ pip install -r requirements-test.txt
$ python -m pytest --numprocesses=auto --dist=loadscope tests
```
