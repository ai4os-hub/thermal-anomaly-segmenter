# TASeg

This folder contains the required code for inference with and training of
the Thermal Anomaly Segmenter. It has the following file structure:

```
├── configurations      <- Folder containing basic model configurations
│   ├── train_segformer_b0.json         <- SegFormer B0 (smallest variant)
│   ├── train_segformer_b1.json         <- SegFormer B1
│   ├── train_segformer_b2.json         <- SegFormer B2 (best performing!)
│   ├── train_segformer_b3.json         <- SegFormer B3
│   ├── train_segformer_b4.json         <- SegFormer B4
│   ├── train_smp_deeplabV3plus.json    <- DeepLabV3+
│   ├── train_smp_pspnet.json           <- PSPNet
│   └── train_smp_unet.json             <- UNet
├── data                <- Folder containing scripts for data preparation
│   ├── __init__.py                 <- Package import entrypoint
│   ├── augmentation.py             <- Module for data augmentation
│   ├── dataset_creation.py         <- Module for dataset creation (unused)
│   └── dataset.py                  <- Module for dataset processing / loading
├── helpers             <- Folder containing specific helper functions
│   ├── __init__.py                 <- Package import entrypoint
│   ├── basic.py                    <- Module for basic helper functions
│   ├── torch_inference.py          <- Module for inference helper functions
│   └── vignetting_correction.py    <- Module for data helper functions
├── models              <- Folder containing scripts for data preparation
│   ├── __init__.py                 <- Package import entrypoint
│   ├── segformer_model.py          <- Module for segformer model
│   └── smp_model.py                <- Module for smp model
│
├── __init__.py         <- Package import entrypoint, predict and train coordination
├── config.py           <- Module for CONSTANTS shared between files
├── prediction.py       <- Script for performing prediction
├── training.py         <- Script for performing training
└── utils.py            <- Module for general utility functions
```

The arguments used for the here defined `prediction.py` and `training.py`
come from the `api/schemas.py`, the values of which are provided by user
inputs. Alternatively, you can run both directly via CL. For more information,
see the [main README.md](../README.md).
