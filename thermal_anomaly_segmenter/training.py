"""
Script for coordinating training of semantic segmentation models
"""
from datetime import datetime
import gc
import json
import logging
import os
from pathlib import Path
import random
import mlflow

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from thermal_anomaly_segmenter import config as cfg
from thermal_anomaly_segmenter.models.segformer_model import (
    SegFormer, HfModelCheckpoint, load_segformer_model
)
from thermal_anomaly_segmenter.models.smp_model import SMPModel
from thermal_anomaly_segmenter.utils import (
    load_config, update_config
)
from thermal_anomaly_segmenter.data.dataset import SegmentationDataset
from thermal_anomaly_segmenter.data.augmentation import (
    get_training_augmentation_thermal
)

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


def train_segmentation(**kwargs):
    """
    Train a segmentation model using the given config

    :param train_config_path:  path to the config file defining training
    :return: trained model stored to config output path
    """

    # Load config file and update with user inputs
    train_config_path = Path(
        cfg.BASE_PATH, "thermal_anomaly_segmenter", "configurations",
        "train_" + kwargs["model_type"] + ".json"
    )
    train_config = load_config(train_config_path)
    train_config = update_config(train_config, **kwargs)
    logger.info(f"Training with config:\n{train_config}")

    # ------------------ Define parameters
    batch_size = train_config.get("batch_size")
    gpus = train_config.get("gpus")
    worker_count = train_config.get("worker_count")
    model_type = train_config.get("model")

    run_name = train_config.get("config_name") + "_" \
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    expanded_output_path = os.path.expandvars(train_config["output_path"])
    output_path = os.path.join(expanded_output_path, run_name)
    os.makedirs(output_path, exist_ok=True)

    # Store updated training config to output_path
    with open(os.path.join(output_path, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    # Set seed for reproducibility
    random.seed(kwargs["seed"])
    torch.manual_seed(kwargs["seed"])
    pl.seed_everything(kwargs["seed"])

    # ------------------- Load data
    datasets = load_datasets(train_config)

    # ------------------- Load & initialise model
    logger.info("Loading model...")

    main_dataset = datasets[train_config["main_dataset"]]
    mean = main_dataset["mean"]     # for normalisation
    std = main_dataset["std"]       # for normalisation

    if model_type == "segformer":
        pretrained_weights = train_config.get("pretrained_weights")
        model = SegFormer(
            mean=mean, std=std,
            segformer_pretrained_model=pretrained_weights,
            id2label=main_dataset["train_dataset"].id2label,
            lr_scheduler=train_config.get("lr_scheduler"),
            loss_fn=train_config.get("loss_fn")
        )

    elif model_type == "smp":
        import ssl

        # Required to ensure weights download despite SSL certificate issues
        ssl._create_default_https_context = ssl._create_unverified_context

        smp_encoder = train_config.get("smp_encoder", "resnet101")
        smp_arch = train_config.get("smp_arch", "DeepLabV3Plus")

        model = SMPModel(
            mean=mean, std=std,
            encoder_name=smp_encoder,
            arch=smp_arch,
            lr_scheduler=train_config.get("lr_scheduler"),
            loss_fn=train_config.get("loss_fn")
        )

    else:
        raise ValueError(f"Unknown model type {model_type}")

    # ----------------- Run training
    logger.info("Starting training...")
    for train_phase_name in train_config["train_phases"]:
        logger.info(f"Starting train phase: {train_phase_name}")
        train_phase = train_config["train_phases"][train_phase_name]

        train_dataset = datasets[train_phase["dataset"]]["train_dataset"]
        val_dataset = datasets[train_phase["dataset"]]["val_dataset"]
        test_dataset = datasets[train_phase["dataset"]]["test_dataset"]

        train_dataloader = DataLoader(
            train_dataset, batch_size, shuffle=True, num_workers=worker_count
        )
        valid_dataloader = DataLoader(
            val_dataset, batch_size, shuffle=False, num_workers=worker_count
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size, shuffle=False, num_workers=worker_count
        )

        # Freeze or unfreeze encoder depending on model type
        if model_type == "segformer":
            if train_phase["freeze_encoder"]:
                for param in model.model.segformer.encoder.parameters():
                    param.requires_grad = False
            else:
                for param in model.model.segformer.encoder.parameters():
                    param.requires_grad = True
        else:
            if train_phase["freeze_encoder"]:
                for param in model.model.encoder.parameters():
                    param.requires_grad = False
            else:
                for param in model.model.encoder.parameters():
                    param.requires_grad = True

        # Create custom checkpoint callback
        # (necessary for saving model checkpoints of huggingface models)
        checkpoint_output_path = os.path.join(
            output_path, "checkpoints", train_phase_name
        )

        if model_type == "segformer":
            # This custom checkpoint callback makes sure that the model
            # is saved in the correct format and transfomer outputs
            checkpoint_callback = HfModelCheckpoint(
                dirpath=checkpoint_output_path,
                monitor='valid_dataset_iou',
                mode='max',
                save_last=True,
                save_top_k=1,
            )
        else:
            # For regular models use the default checkpoint callback
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_output_path,
                monitor='valid_dataset_iou',
                mode='max',
                save_last=True,
                save_top_k=1,
            )

        # Create logging (CSV always, MLFlow if user provided)
        logger.info("Setting up model logging...")
        logger_list = []

        csv_logger = CSVLogger(save_dir=output_path,
                               name="csv_log")
        logger_list.append(csv_logger)

        if kwargs["mlflow"]:
            mlflow.set_tracking_uri(cfg.MLFLOW_URI)
            mlflow.set_experiment(experiment_name=cfg.MLFLOW_EXPERIMENT_NAME)

            mlflow_logger = MLFlowLogger(
                experiment_name=cfg.MLFLOW_EXPERIMENT_NAME,
                run_name=f"{run_name}__{train_phase_name}"
            )
            mlflow_logger.experiment.set_tag(
                mlflow_logger.run_id, "group_run_name", run_name
            )
            logger_list.append(mlflow_logger)

        # Monitor the current learning rate and log it
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

        # -------------- Create trainer with given parameters
        logger.info("Creating pytorch lightning Trainer...")
        model.lr = train_phase["lr"]   # set lr
        trainer = pl.Trainer(
            max_epochs=train_phase["epochs"],
            callbacks=[checkpoint_callback, lr_monitor],
            logger=logger_list,
            accelerator="gpu",
            devices=gpus,
            precision="16-mixed"
        )

        # -------------- Fit model (start training)
        logger.info("Training model...")
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        logger.info("Evaluating last model on test set...")
        test_result = trainer.test(
            model,
            dataloaders=[test_dataloader]
        )
        logger.info(f"Test results:\n{test_result}")

        logger.info("Loading best model...")
        if model_type == "segformer":
            best_model = load_segformer_model(
                checkpoint_callback.best_model_path
            )
        else:
            best_model = SMPModel.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )

        logger.info("Evaluating best model on val and test sets...")
        val_best_result = trainer.validate(
            best_model,
            dataloaders=[valid_dataloader]
        )
        test_best_result = trainer.test(
            best_model,
            dataloaders=[test_dataloader]
        )
        logger.info(f"Test results:\n{val_best_result}")
        logger.info(f"Test results:\n{test_best_result}")

        # Print free gpu memory after training on all available gpus
        if gpus >= 1:
            logger.info("Get available gpus from torch...")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.memory_allocated(i)} / "
                            f"{torch.cuda.max_memory_allocated(i)}")

        trainer = None
        gc.collect()    # Trigger python garbage collection
        torch.cuda.empty_cache()     # Free GPU memory

        # Print free gpu memory after freeing memory.
        if gpus >= 1:
            logger.info("Memory after clearing gpu memory...")
            available_gpus = torch.cuda.device_count()
            for i in range(available_gpus):
                logger.info(f"GPU {i}: {torch.cuda.memory_allocated(i)} / "
                            f"{torch.cuda.max_memory_allocated(i)}")

    return output_path


def load_datasets(train_config):
    """
    Load the data defined in the given train_config

    :param train_config: config defining the train process
    :return: dictionary with the loaded data
    """

    logger.info("Loading datasets...")
    datasets = {}

    for name, dataset_config in train_config["datasets"].items():
        if "normalize_on_other_dataset" in dataset_config:
            continue    # skip to load later
        load_dataset(datasets, name, dataset_config)

    # Load skipped data
    for name, dataset_config in train_config["datasets"].items():
        if "normalize_on_other_dataset" in dataset_config:
            load_dataset(datasets, name, dataset_config)

    return datasets


def load_dataset(datasets, name, dataset_config):
    """
    Load the data defined in the given dataset_config and add it to the
    data dictionary

    :param datasets: dictionary with the data that have already been loaded.
      Will be updated with the new data
    :param name: name of the data
    :param dataset_config: config defining the data
    """

    logger.info(f"Loading '{name}' with config:\n{dataset_config}")
    augment_train = dataset_config.get("augment_train")
    augmentation_range = tuple(dataset_config.get("augmentation_range"))

    if augment_train:
        train_augmentation = get_training_augmentation_thermal()
    else:
        train_augmentation = None

    # Expand environment variables
    dataset_path = os.path.expandvars(dataset_config["path"])
    stats_dataset = SegmentationDataset(dataset_path, split="train")
    mean, std = stats_dataset.get_mean_std()

    # Set mean and std to use for normalization
    mean_transform = mean
    std_transform = std
    if "normalize_on_other_dataset" in dataset_config:
        mean_transform = datasets[
            dataset_config["normalize_on_other_dataset"]
        ]["mean"]
        std_transform = datasets[
            dataset_config["normalize_on_other_dataset"]
        ]["std"]

    # Create transforms for data
    short_size = 512
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # Resize to 512x512 and pad if necessary
         transforms.Resize(short_size, antialias=True),
         transforms.Normalize(mean_transform, std_transform)]
    )
    target_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(short_size, antialias=True)]
    )

    # Create data for the different splits
    train_dataset = SegmentationDataset(
        dataset_path, split="train",
        transform=transform,
        target_transform=target_transform,
        augmentation=train_augmentation,
        augmentation_range=augmentation_range,
        pixel_threshold=40,
        repeat_images=dataset_config.get("repeat_train_images", 1),
    )
    val_dataset = SegmentationDataset(
        dataset_path, split="val",
        transform=transform,
        target_transform=target_transform
    )
    test_dataset = SegmentationDataset(
        dataset_path, split="test",
        transform=transform,
        target_transform=target_transform
    )
    # Add data to data dictionary
    datasets[name] = {"mean": mean,
                      "std": std,
                      "train_dataset": train_dataset,
                      "val_dataset": val_dataset,
                      "test_dataset": test_dataset}
