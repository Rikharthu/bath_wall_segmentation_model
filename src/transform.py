import numpy as np
import albumentations as A
from src import config
import segmentation_models_pytorch as smp


def image_to_tensor_format(x, **_kwargs):
    # HWC -> CHW
    return x.transpose(2, 0, 1).astype(np.float32)


def mask_to_tensor_format(x, **_kwargs):
    return np.expand_dims(x, 0).astype(np.float32)


def get_preprocessing_transform(preprocessing_fn):
    if isinstance(preprocessing_fn, str):
        preprocessing_fn = smp.encoders.get_preprocessing_fn(preprocessing_fn)

    return A.Compose([
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=image_to_tensor_format, mask=mask_to_tensor_format)
    ])


def get_train_augmentations():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=15,
            shift_limit=0.1,
            p=1,
            border_mode=0
        ),
        A.Perspective(p=0.5),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightness(p=1),
            A.RandomGamma(p=1),
        ], p=0.9),
        A.OneOf([
            A.Sharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        A.OneOf([
            A.RandomContrast(p=1),
            A.HueSaturationValue(p=1),
        ]),
        A.LongestMaxSize(max_size=min(config.INPUT_IMAGE_SIZE), interpolation=1),
        A.PadIfNeeded(
            min_height=config.INPUT_IMAGE_SIZE[1],
            min_width=config.INPUT_IMAGE_SIZE[0],
            border_mode=0
        ),
    ])


def get_val_augmentations():
    return A.Compose([
        A.LongestMaxSize(max_size=min(config.INPUT_IMAGE_SIZE), interpolation=1),
        A.PadIfNeeded(
            min_height=config.INPUT_IMAGE_SIZE[1],
            min_width=config.INPUT_IMAGE_SIZE[0],
            border_mode=0
        ),
    ])