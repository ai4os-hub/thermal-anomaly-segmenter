"""
Script for coordinating inference with a trained segmentation model.
"""
import os
import io
from pathlib import Path
import logging
import time

import cv2
import matplotlib.pyplot as plt
import lz4.frame
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

from thermal_anomaly_segmenter import utils, config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

if torch.cuda.is_available():
    freest_gpu = utils.get_freest_gpu()
    DEVICE = f"cuda:{freest_gpu}" if freest_gpu is not None else "cuda:0"
else:
    DEVICE = "cpu"


def infer_on_npy(
        model_dir: Path,
        npy_path: Path,
        threshold: float = 0.5,
):
    """
    Run inference on a single image

    :param model_dir: Path to model directory containing a train_config.json
    :param npy_path: Path to image file (*.npy.lz4) to infer upon
    :param threshold: Threshold to use for inference
    """
    logger.info(f"Using device: {DEVICE}")

    # Get model config
    config_path = Path(model_dir, 'train_config.json')
    if not config_path.is_file():
        raise FileNotFoundError(f"No config found at '{config_path}'.")

    # Create "predictions" subfolder in model directory to save results to
    dst_dir = os.path.join(model_dir, "predictions")
    os.makedirs(dst_dir, exist_ok=True)

    # Load config from config_path and make it available as a dict
    cfg_dict = utils.load_config(config_path)
    model_type = cfg_dict["model"]
    try:
        ckpt_path = sorted(model_dir.rglob("epoch=*.ckpt"))[0]
    except IndexError:
        raise FileNotFoundError(f"No checkpoint found in '{model_dir}'.")

    start = time.time()
    # Load model
    model = load_model(model_type=model_type, ckpt_path=ckpt_path)
    mod_time = time.time()

    # Load and prepare image data
    img, mask = load_img(str(npy_path))
    inp = prep_data(inp=img, model=model)
    img_time = time.time()

    # Perform inference
    pred_img = run_inference(model=model, data=inp, threshold=threshold)

    end = time.time()
    logger.info(f"Time to load model: {round(mod_time - start, 3)} seconds")
    logger.info(f"Time to load image: {round(img_time - mod_time, 3)} seconds")
    logger.info(f"Time for inference: {round(end - img_time, 3)} seconds")
    logger.info(f"TOTAL TIME: {round(end - start, 3)} seconds")

    # Save results as .npy and .png
    filename = utils.get_filestem(npy_path)
    filepath = os.path.join(dst_dir, filename + "_" + str(threshold))
    np.save(filepath + ".npy", pred_img)

    pred_img_greyscale = (pred_img * 255).astype(np.uint8)
    cv2.imwrite(filepath + ".png", pred_img_greyscale)

    pred_img_masked = np.ma.masked_where(mask, pred_img)
    plt.imshow(pred_img_masked, cmap="gray")
    plt.savefig(filepath + "_masked.png")
    plt.close()


def load_model(model_type, ckpt_path):
    """
    Load model from checkpoint path depending on model type

    :param model_type: type of model to use (segformer or smp)
    :param ckpt_path: path to model checkpoint
    :return: model
    """

    if model_type == "segformer":
        from thermal_anomaly_segmenter.models.segformer_model import (
            load_segformer_model
        )

        model = load_segformer_model(
            checkpoint_path=str(ckpt_path)
        ).to(DEVICE).eval()
        return model

    elif model_type == "smp":
        from thermal_anomaly_segmenter.models.smp_model import SMPModel

        model = SMPModel.load_from_checkpoint(
            checkpoint_path=str(ckpt_path)
        ).to(DEVICE).eval()
        return model

    else:
        raise ValueError("Unknown model type:", model_type)


def load_img(npy_path: str):
    """
    Load image from numpy file and extract mask

    :param npy_path: Path to numpy file
    :return: loaded image (3-ch), extracted mask (1-ch)
    """
    with open(npy_path, "rb") as f:
        compressed_data = f.read()
        uncompressed_data = lz4.frame.decompress(compressed_data)
        file_like_object = io.BytesIO(uncompressed_data)
        img = np.load(file_like_object)
        mask = get_mask(img[:, :, 0])

    return img, mask


def get_mask(T_arr):
    """
    Get a boolean mask indicating which pixels are in the unmasked region

    :param T_arr: 1D temperature array masked to the pipline area
    :return: boolean mask (single channel)
    """
    mask_limit = np.percentile(T_arr, 0.99)
    mask = np.where(T_arr <= mask_limit, True, False)

    return mask


def prep_data(inp, model):
    """Prepare data for inference

    :param inp: image input data to prepare
    :param model: model with which to calculate transform (normalisation)
    """
    tensor_inp = tensorize(inp)
    norm_inp = normalize(tensor_inp, model).unsqueeze(0).to(DEVICE)
    return norm_inp


def normalize(inp: torch.Tensor, model) -> torch.Tensor:
    """Normalise the provided tensor using model mean and std"""
    transform = transforms.Normalize(mean=model.mean, std=model.std)
    return transform(inp)


def tensorize(inp: np.ndarray) -> torch.Tensor:
    """Turn numpy array into a tensor"""
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(inp)


def run_inference(model, data, threshold):
    """Run inference

    :param model: model to run inference with
    :param data: data to run inference on
    :param threshold: threshold to use for inference
    :return: prediction mask
    """
    try:
        with torch.no_grad():
            logits = model(data).logits

    except AttributeError:
        raise Exception(
            "Selected model not capable of inference "
            "(probably due to interrupted training)!"
        )

    out = nn.functional.interpolate(
        logits,
        size=data.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    # Get mask outputs
    prob = out.sigmoid()  # Sigmoid to get probabilities

    # Threshold
    binary_mask = (prob > threshold).float()
    return binary_mask.squeeze().cpu().numpy()  # Convert to numpy
