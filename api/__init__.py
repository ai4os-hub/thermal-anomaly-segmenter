"""Endpoint functions to integrate your model with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/ai4os-hub/demo-advanced
"""
import logging
from pathlib import Path
import shutil

import thermal_anomaly_segmenter

from . import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("GET 'metadata' called. Collected data from: %s",
                    config.API_NAME)
        metadata = {**config.PROJECT_METADATA, **{
            "datasets_local": utils.get_dirs(
                config.DATA_PATH, patterns={'manual_set', 'generated_set'}
            ),
            "datasets_remote": utils.get_dirs(
                config.REMOTE_PATH, patterns={'manual_set', 'generated_set'}
            ),
            "models_local": utils.get_dirs(
                config.MODELS_PATH, patterns={"train_config.json"}
            ),
            "models_remote": utils.get_dirs(
                config.REMOTE_PATH, patterns={"train_config.json"}
            ),
        }}  # Combine general metadata with data and model locations
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(
        accept='application/json',
        **options
):
    """Performs model prediction from given input data and parameters.

    Arguments:
        accept -- Response parser type, default is json
        **options -- All other user input arguments from PredArgsSchema

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values (dict or str) or files.
    """

    try:  # Call AI model predict() method
        # handle input file (path from a browsing webargs field)
        input_filename = options['input_file'].original_filename
        if not input_filename.endswith(".npy.lz4"):
            raise ValueError(
                f"Selected input '{input_filename}' is not a .npy.lz4!"
            )
        # copy from temporary location to DATA_PATH
        tmp_filepath = Path(options['input_file'].filename)
        input_filepath = Path(config.DATA_PATH, input_filename)
        shutil.copy(tmp_filepath, input_filepath)
        options = {**{"input_filepath": input_filepath}, **options}

        # run prediction
        for k, v in options.items():
            logger.info(f"POST 'predict' argument - {k}:\t{v}")

        result = thermal_anomaly_segmenter.predict(**options)

        # delete copied image
        input_filepath.unlink()

        logger.info("Predict result: %s", result)
        logger.info("Returning content_type for: %s", accept)
        return responses.content_types[accept](result, **options)
    except Exception as err:
        logger.error("Error while running POST 'predict': %s",
                     err, exc_info=True)
        raise  # Reraise the exception after log


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**options):
    """Performs model training from given input data and parameters.

    Arguments:
        **options -- User input arguments from TrainArgsSchema

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        Parsed history/summary of the training process.
    """

    try:  # Call your AI model train() method

        # MLFlow experiment tracking requires setting environment variables
        # and getting/injecting necessary credentials
        if options["mlflow"]:
            try:
                logger.info(
                    f"MLFlow model experiment tracking "
                    f"on {config.MLFLOW_URI}"
                    f"for user {config.MLFLOW_USERNAME}"
                )
                utils.check_mlflow_authentication()
                logger.info('MLFlow configuration complete.')
            except Exception:
                options["mlflow"] = False

        for k, v in options.items():
            logger.info(f"POST 'train' argument - {k}:\t{v}")

        result = thermal_anomaly_segmenter.train(**options)

        logger.info(f"POST 'train' result: {result}")
        return result
    except Exception as err:
        logger.error("Error while running 'POST' train: %s",
                     err, exc_info=True)
        raise  # Reraise the exception after log
