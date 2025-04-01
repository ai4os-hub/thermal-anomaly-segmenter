"""Module for defining custom web fields to use on the API interface.
This module is used by the API server to generate the input form for the
prediction and training methods. You can use any of the defined schemas
to add new inputs to your API.

The module shows simple but efficient example schemas. However, you may
need to modify them for your needs.
"""
from pathlib import Path
import marshmallow
from webargs import ValidationError, fields, validate
from torch.cuda import is_available

from . import config, responses, utils


class NpyFile(fields.String):
    """Field that takes a file path as a string and makes sure it exists
    either locally in repository directory or remotely on Nextcloud,
    whilst also ensuring it's a numpy file.
    """
    def _deserialize(self, value, attr, data, **kwargs):
        if Path(value).is_file():
            if value.endswith(".npy.lz4"):
                return value
            raise ValidationError(
                f"Provided file path `{value}` is not a .npy.lz4 file."
            )
        else:
            raise ValidationError(
                f"Provided file path `{value}` does not exist."
            )


class Directory(fields.String):
    """Field that takes a directory as a string and makes sure it exists.
    """
    def deserialize(self, value, attr, data, **kwargs):
        if value is not marshmallow.missing:
            if Path(value).is_dir():
                return value
            else:
                raise ValidationError(
                    f"Provided `{value}` not an existing directory!"
                )
        else:
            raise ValidationError(
                f"No {attr} was defined, but this parameter is required!"
            )


class Dataset(fields.String):
    """Field that takes a string and validates against current available
    data files at config.DATA_PATH.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if value not in utils.ls_dirs(config.DATA_PATH):
            raise ValidationError(f"Dataset `{value}` not found.")
        return str(config.DATA_PATH / value)


class ForceTrue(fields.Boolean):
    """Field that takes a boolean and makes sure it's true."""

    def _deserialize(self, value, attr, data, **kwargs):
        if not value:
            raise ValidationError(f"{attr} must be True!")
        return value


class PredArgsSchema(marshmallow.Schema):
    """Prediction arguments schema for api.predict function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    input_file = fields.Field(
        required=True,
        metadata={
            'type': "file",
            'location': "form",
            'description': 'Input a 3-channel .npy file to infer upon.'
        }
    )

    # Helper variables to avoid long f-strings
    local_dirs = utils.get_dirs(config.MODELS_PATH,
                                patterns={"train_config.json"})
    remote_dirs = utils.get_dirs(config.REMOTE_PATH,
                                 patterns={"train_config.json"})

    model_dir = Directory(
        metadata={
            'description': (
                'Model to be used for prediction. Results will be saved '
                'to a "predictions" folder in the selected model directory.'
                '\n\nCurrently existing model paths are:'
                f'\n- local:\n{local_dirs}'
                f'\n- remote:\n{remote_dirs}\n'
            )
        },
        load_default=utils.get_default_path(
            directory_list=[config.MODELS_PATH, config.REMOTE_PATH],
            patterns={"train_config.json"}),
    )

    threshold = fields.Float(
        metadata={
            'description': "Threshold for prediction."
        },
        validate=validate.Range(min=0.0, max=1.0),
        load_default=0.5,
    )

    accept = fields.String(
        metadata={
            "description": "Return format for method response.",
            "location": "headers",
        },
        validate=validate.OneOf(list(responses.content_types)),
        load_default='application/json',
    )


class TrainArgsSchema(marshmallow.Schema):
    """Training arguments schema for api.train function."""

    class Meta:  # Keep order of the parameters as they are defined.
        # pylint: disable=missing-class-docstring
        # pylint: disable=too-few-public-methods
        ordered = True

    model_type = fields.String(
        metadata={
            "description": "Type of semantic segmentation model to train.",
        },
        validate=validate.OneOf(sorted(config.MODEL_VARIANTS.keys())),
        load_default=sorted(config.MODEL_VARIANTS.keys())[0],
    )

    train_phases = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Training phases. Options are:\n"
                "1. 'pre_train_frozen': train on generated set, weight frozen"
                "\n2. 'pre_train_unfrozen': train on generated set, weight "
                "unfrozen\n3. 'fine_tune': fine tune on manual set, weights "
                "unfrozen\n\nFor best performance, use all three. "
                "In case of CUDA_OUT_OF_MEMORY error, only fine tune."
            )
        },
        validate=validate.ContainsOnly(config.TRAIN_PHASES),
        load_default=list(config.TRAIN_PHASES),
    )

    # Helper variables to avoid long f-strings
    list_of_patterns = [{'manual_set', 'generated_set'},
                        {'manual_set.zip', 'generated_set.zip'}]
    local_dirs = utils.get_dirs_(config.DATA_PATH, list_of_patterns)
    remote_dirs = utils.get_dirs_(config.REMOTE_PATH, list_of_patterns)

    dataset_path = Directory(
        metadata={
            'description':
                'Path to the dataset.\n\nAvailable paths are:\n'
                f'- local: {local_dirs}\n'
                f'- remote: {remote_dirs}'
        },
        load_default=utils.get_default_path(
            directory_list=[config.REMOTE_PATH, config.DATA_PATH],
            patterns=list_of_patterns
        ),
    )

    data_augmentation = fields.Boolean(
        metadata={
            "description": "Implement data augmentation for training?",
        },
        load_default=True,
    )

    loss_fn = fields.String(
        metadata={
            "description": "Loss function for training.",
        },
        validate=validate.OneOf(sorted(config.LOSS_FUNCTIONS.keys())),
        load_default=list(config.LOSS_FUNCTIONS.keys())[0]
    )

    batch_size = fields.Integer(
        metadata={
            "description": (
                "Batch size.\nFor larger models (i.e. SegFormer B4), "
                "smaller values (i.e. 8) are better.\nTo prevent "
                "CUDA_OUT_OF_MEMORY error, define even smaller values."
            )
        },
        validate=validate.Range(min=1),  # minimum value has to be 1
        load_default=16,
    )

    worker_count = fields.Integer(
        metadata={
            'description': (
                'Worker count.\nProposed default value '
                'is based on current deployment setup.\n'
                'NOTE: Deployments may slow down if value is too high!'
            )
        },
        validate=validate.Range(min=1),  # minimum value has to be 1
        load_default=utils.get_optimal_num_workers(),
    )

    seed = fields.Integer(
        metadata={'description': 'Seed for weight initilisation.'},
        validate=validate.Range(min=1),  # minimum value has to be 1
        load_default=10,
    )

    gpu = ForceTrue(
        metadata={'description': 'Is GPU available for training?'},
        load_default=is_available(),
    )

    mlflow = fields.Boolean(
        metadata={
            "description": "Use MLFlow for experiment tracking?",
        },
        load_default=True,
    )
