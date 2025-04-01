"""Package to create dataset, build training and prediction pipelines.

This file should define or import all the functions needed to operate the
methods defined at thermal_anomaly_segmenter/api.py. Complete the TODOs
with your own code or replace them importing your own functions.
For example:
```py
from your_module import your_function as predict
from your_module import your_function as training
```
"""
import logging
from pathlib import Path

from thermal_anomaly_segmenter import config, utils
from thermal_anomaly_segmenter.prediction import infer_on_npy
from thermal_anomaly_segmenter.training import train_segmentation

logger = logging.getLogger(__name__)
utils.configure_logging(logger, config.LOG_LEVEL)


class ResultError(Exception):
    """Error for missing result"""
    pass


def predict(**options):
    """Main/public method to perform prediction.
    Saves the prediction results to a "predictions" folder in the model
    directory, wherever it is located (remote or local).
    """

    model_dir = Path(options["model_dir"])
    input_filepath = Path(options["input_filepath"])

    infer_on_npy(
        model_dir=model_dir,
        npy_path=input_filepath,
        threshold=options["threshold"],
    )

    # return prediction results
    if Path(model_dir, 'predictions').is_dir():
        pred_results = [
            f for f in Path(model_dir, 'predictions').rglob("*.png")
            if utils.get_filestem(Path(input_filepath)) in f.stem
        ]

        if pred_results:
            predict_result = {
                'result': f'{pred_results[0]}'
            }
        else:
            predict_result = {
                'result': f"Error occurred. No matching prediction "
                          f"results for file '{input_filepath}' "
                          f"in '{model_dir}'."
            }
    else:
        predict_result = {
            'result': f"Error occurred. No prediction folder "
                      f"created at '{model_dir}'."
        }
    logger.debug(f"[predict()]: {predict_result}")

    return predict_result


def train(**options):
    """Main/public method to perform training.
    Saves trained model to a config.MODELS_PATH folder.
    """

    # If data is zipped, unzip into given folder
    zip_paths = list(Path(options["dataset_path"]).glob("*.zip"))
    if zip_paths:
        logger.info(f"Extracting data from {len(zip_paths)} .zip files...")
        utils.unzip(zip_paths)

        for z in zip_paths:
            utils.check_path_exists(Path(z.parent, z.stem))

    # Perform training
    try:
        model_dir = train_segmentation(**options)
    except Exception as err:
        logger.error("Error while training: %s", err, exc_info=True)
        raise ResultError(
            f"Error during model training: {err}"
        )

    # Return logged model results
    try:
        csv_log_path = sorted(Path(model_dir).rglob("metrics.csv"))[-1]
        results_dict = utils.extract_log_dict(csv_log_path)
        return results_dict

    except IndexError as e:
        raise ResultError(
            f"Error during training, no metrics.csv found in model directory "
            f"'{model_dir}'!", e
        )
