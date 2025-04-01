"""
Helper functions for creating data as required for semantic segmentation
"""
import io
import json
import os
import pickle  # nosec B403

import cv2
import lz4
import numpy as np
from matplotlib import pyplot as plt

from helpers.basic import get_mask


def create_dl_dataset(
        output_path,
        dhs_dataset_dict,
        segmentation_results,
        masked,
        image_type,
        train_split: list = ["train"],
        val_split: list = ["val"],
        test_split: list = ["test"],
        channel_mode: str = "mixed",
        split_config_path=None,
        label_type: str = "pickle"
):
    """
    Create data for training the deep learning semantic segmentation models

    :param output_path: path to output directory
    :param dhs_dataset_dict: dict containing all required DHSDataset objects
    :param segmentation_results: path to the segmentation results / labels
    :param masked: determines weather to use masked or unmasked images
    :param image_type: determines weather to use thermal or rgb images
    :param train_split: name of the "splits" to be included in the training set
    :param val_split: name of the "splits" to be included in the validation set
    :param test_split: name of the "splits" to be included in the test set
    :param channel_mode: determines how the channels are handled
      (repeat, mixed, single). Here: mixed.
    :param split_config_path: path to a json file containing the split
      configuration. If None, all images are used.
    :param label_type: type of label to use (pickle or png)
    :return: Dataset
    """

    # Convert dict keys to list of strings
    dataset_keys = list(dhs_dataset_dict.keys())
    dataset_path = output_path

    # If split_config_path is None, subsets is equal to dataset_keys
    if split_config_path is None:
        subset_names = dataset_keys
    else:
        subset_names = ["train", "val", "test"]

    # Create data info dict
    dataset_info = {
        "image_type": image_type,
        "masked": masked,
        "subsets": subset_names,
        "labels": ["background", "thermal"],
        "default_splits": {
            "train": train_split,
            "val": val_split,
            "test": test_split
        }
    }

    # If channel_mode is not None, add it to dataset_info
    if channel_mode is not None:
        dataset_info["channel_mode"] = channel_mode

    # Create data directory if it does not exist
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    else:
        raise Exception("Dataset directory already exists!")

    # Create data info file as json
    with open(os.path.join(dataset_path, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)

    # Create subsets dict
    subsets = {}
    if split_config_path is None:

        # Iterate over data in dhs_dataset_dict
        for dataset_name, dataset in dhs_dataset_dict.items():
            subsets[dataset_name] = []
            # subset_path = os.path.join(dataset_path, dataset_name)

            image_path = os.path.join(dataset_path, "image", dataset_name)
            label_path = os.path.join(dataset_path, "label", dataset_name)
            preview_path = os.path.join(dataset_path, "preview", dataset_name)

            # Create subfolder image in subset_path if it does not exist
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            # Create subfolder label in subset_path if it does not exist
            if not os.path.exists(label_path):
                os.makedirs(label_path)

            # Create subfolder preview in subset_path if it does not exist
            if not os.path.exists(preview_path):
                os.makedirs(preview_path)

            # Replace "{dataset_name}" in segm_results_path with dataset_name
            segm_results_path = segmentation_results.replace(
                "dataset_name", dataset_name
            )

            # Get pickle files in segm_results_path
            pickle_files = os.listdir(segm_results_path)

            # Iterate over pickle files
            for pickle_file in pickle_files:
                base_file_name = os.path.splitext(pickle_file)[0]
                # If file ending is not .pickle, skip
                if not pickle_file.endswith(".pickle"):
                    continue

                pickle_file_path = os.path.join(segm_results_path, pickle_file)

                subsets[dataset_name].append({
                    "image_name": base_file_name,
                    "dataset_name": dataset_name,
                    "image_identifier": dataset_name + "_" + base_file_name,
                    "label_type": "pickle",
                    "label_file_path": pickle_file_path,
                })
    else:
        split_config = json.load(open(split_config_path, "r"))

        for subset_name in subset_names:
            subsets[subset_name] = []

            image_path = os.path.join(dataset_path, "image", subset_name)
            label_path = os.path.join(dataset_path, "label", subset_name)
            preview_path = os.path.join(dataset_path, "preview", subset_name)

            # Create subfolder image in subset_path if it does not exist
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            # Create subfolder label in subset_path if it does not exist
            if not os.path.exists(label_path):
                os.makedirs(label_path)

            # Create subfolder preview in subset_path if it does not exist
            if not os.path.exists(preview_path):
                os.makedirs(preview_path)

            image_tuples = split_config[subset_name]
            for (dataset_name, image_name) in image_tuples:
                image_identifier = dataset_name + "_" + image_name
                label_sub_path = segmentation_results.replace(
                    "dataset_name", dataset_name
                )

                if label_type == "pickle":
                    label_path = os.path.join(label_sub_path,
                                              image_name + ".pickle")
                else:
                    label_path = os.path.join(label_sub_path,
                                              image_name + ".png")

                subsets[subset_name].append({
                    "image_name": image_name,
                    "image_identifier": image_identifier,
                    "dataset_name": dataset_name,
                    "label_type": label_type,
                    "label_file_path": label_path})

    # Iterate over subsets
    for subset_name, subset in subsets.items():
        image_path = os.path.join(dataset_path, "image", subset_name)
        label_path = os.path.join(dataset_path, "label", subset_name)
        preview_path = os.path.join(dataset_path, "preview", subset_name)

        # Iterate over images in subset
        for image_dict in subset:
            label_type = image_dict["label_type"]
            base_file_name = image_dict["image_name"]
            dataset_name = image_dict["dataset_name"]
            label_file_path = image_dict["label_file_path"]

            dataset = dhs_dataset_dict[dataset_name]

            if label_type == "pickle":
                # Get data from pickle file
                with open(label_file_path, "rb") as f:
                    result_data = pickle.load(f)  # nosec B301
            else:
                print(label_file_path)
                # Load image from label_file_path
                segmentation_mask = cv2.imread(label_file_path,
                                               cv2.IMREAD_GRAYSCALE)
                print("Segmentation mask", segmentation_mask.shape,
                      "Label file path", label_file_path)
                result_data = {
                    "segmentation_mask": segmentation_mask
                }

            create_image(
                result_data,
                dataset,
                masked,
                base_file_name,
                image_path,
                label_path,
                preview_path,
                image_type,
                channel_mode,
                image_identifier=image_dict["image_identifier"]
            )


def create_image(result_data, dataset, masked, base_file_name,
                 image_path, label_path, preview_path, image_type,
                 channel_mode, image_identifier):
    """
    Create image from result_data and save it to image_path

    :param result_data: dictionary containing the segmentation mask
    :param dataset: DHSDataset object
    :param masked: determines weather to use masked or unmasked images
    :param base_file_name: base file name of the image
    :param image_path: path to image directory
    :param label_path: path for storing the label files
    :param preview_path: path for storing the preview images
    :param image_type: determines whether to use thermal or rgb images
    :param channel_mode: determines how the channels are handled
      (repeat, mixed, single). Here: mixed.
    :param image_identifier: identifier of the image
    :return:
    """

    preview_image = None
    if masked:
        # Get image from dhs_dataset data
        image_dict = dataset.get_image(base_file_name)

        image = image_dict["masked_image"]
        unmasked_image = image_dict["unmasked_image"]

        preview_image = image.copy()
        mask = get_mask(image)

        # Skit image if image has no unmasked pixels
        if np.sum(~mask) == 0:
            print("Skip image because it has no unmasked pixels!")
            return

        label = result_data["segmentation_mask"]

        # Save label using cv2
        cv2.imwrite(os.path.join(label_path, image_identifier + ".png"), label)

        if image_type == "thermal":
            print("BBB Image thermal", image.shape)

            image = image.reshape(
                image.shape[0], image.shape[1], 1
            ).astype(np.float32)

            unmasked_image = unmasked_image.reshape(
                unmasked_image.shape[0], unmasked_image.shape[1], 1
            ).astype(np.float32)

            if channel_mode == "repeat":
                # Repeat image 3 times along channel axis
                image = np.repeat(image, 3, axis=2)

            elif channel_mode == "mixed":
                print("Image shape before: ", image.shape)
                image = np.repeat(image, 3, axis=2)
                # Add unmasked_image as last channel
                image[:, :, 2] = unmasked_image[:, :, 0]

            elif channel_mode == "single" or channel_mode is None:
                pass

            else:
                raise Exception("Invalid channel mode!")

            file_save_path = os.path.join(image_path,
                                          image_identifier + ".npy.lz4")

            # Save image as .npy
            # Save numpy compressed with lz4
            buffer = io.BytesIO()
            np.save(buffer, image)

            buffer.seek(0)

            # Compress data using lz4
            compressed_data = lz4.frame.compress(buffer.read())

            # Save compressed data to file
            with open(file_save_path, "wb") as f:
                f.write(compressed_data)

        elif image_type == "rgb":
            file_save_path = os.path.join(image_path,
                                          image_identifier + ".png")

            # Save image using matplotlib with colrmap inferno
            plt.imsave(file_save_path, image, cmap="inferno")
        else:
            raise Exception("Invalid image type!")

        # Save preview image
        preview_image_save_path = os.path.join(preview_path,
                                               image_identifier + ".png")
        # Plot T_arr, mask_image side by side and save
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(preview_image, cmap="inferno")
        mask = get_mask(preview_image)
        masked_label = np.ma.masked_where(mask, label)
        ax2.imshow(masked_label)

        plt.savefig(preview_image_save_path)
        plt.close()

    else:
        image_dict = dataset.get_image(base_file_name)

        image = image_dict["unmasked_image"]
        preview_image = image.copy()
        label = result_data["unmasked_segmentation_mask"]
        # Save label using cv2
        cv2.imwrite(os.path.join(label_path, image_identifier + ".png"), label)

        if image_type == "thermal":
            file_save_path = os.path.join(image_path,
                                          image_identifier + ".npy.lz4")
            image = image.reshape(
                image.shape[0], image.shape[1], 1
            ).astype(np.float32)

            if channel_mode == "repeat":
                # Repeat image 3 times along channel axis
                image = np.repeat(image, 3, axis=2)
            elif channel_mode == "single" or channel_mode is None:
                pass
            else:
                raise Exception("Invalid channel mode!")

            # Save image as .npy lz4 compressed
            buffer = io.BytesIO()
            np.save(buffer, image)

            buffer.seek(0)
            np.save(file_save_path, buffer.getvalue())

        elif image_type == "rgb":
            file_save_path = os.path.join(image_path, base_file_name + ".png")

            # Save image using matplotlib with colormap inferno
            plt.imsave(file_save_path, image, cmap="gray")

        # Save preview image
        preview_image_save_path = os.path.join(preview_path,
                                               base_file_name + ".png")
        # Plot T_arr, mask_image side by side and save
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(preview_image, cmap="gray")
        ax2.imshow(label)
        plt.savefig(preview_image_save_path)
        plt.close()
