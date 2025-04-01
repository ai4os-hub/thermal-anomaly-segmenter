"""
Helper functions for data augmentation transform
"""
import albumentations as albu
import torchvision


def pad_to_square(img, fill=0):
    """
    Pad image to square shape

    :param img: image to pad
    :param fill: padding value
    :return: padded image (long side x long side)
    """
    w, h = img.size
    max_size = max(w, h)
    hp = int((max_size - w) / 2)
    vp = int((max_size - h) / 2)
    hp2 = max_size - w - hp
    vp2 = max_size - h - vp

    padding = (hp, vp, hp2, vp2)
    return torchvision.transforms.functional.pad(
        img, padding, fill=fill, padding_mode='constant'
    )


def get_training_augmentation_thermal():
    """
    Get training augmentation for thermal images

    :return: albumentations transform
    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.3, rotate_limit=40,
                              shift_limit=0.1, p=0.5, border_mode=0,
                              value=-10),
        albu.ElasticTransform(p=0.15),
    ]
    return albu.Compose(train_transform)
