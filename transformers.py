import imgaug.augmenters as iaa
import torch
from PIL import Image
from torchvision.transforms import Compose, FiveCrop, Lambda, RandomCrop, Resize, \
    ToPILImage, ToTensor

from dataset import Denoising, ImgaugWrapper

__all__ = [
    "get_train_transformers",
    "get_test_transformers"
]


def get_train_transformers():
    train_augmenters = iaa.Sequential([
        iaa.Fliplr(p=0.2),
        iaa.Affine(
            translate_px=(-10, 10),
            rotate=(-10, 10),
            mode=["reflect", "symmetric"]
        ),
        iaa.ElasticTransformation(
            alpha=(10, 30),
            sigma=6,
            mode="wrap"
        )
    ], random_order=True)

    transforms = Compose([
        Denoising(denoising_scale=7),
        ImgaugWrapper(train_augmenters),
        ToPILImage(mode="L"),
        RandomCrop(128, pad_if_needed=True),
        Resize(128, interpolation=Image.LANCZOS),
        ToTensor()
    ])
    return transforms


def get_test_transformers():
    for_the_crop = Compose([
        Resize(128, interpolation=Image.LANCZOS),
        ToTensor()
    ])
    return Compose([
        Denoising(denoising_scale=7),
        ToPILImage(mode="L"),
        FiveCrop(128),
        Lambda(lambda crops: torch.stack(tuple([
            for_the_crop(crop) for crop in crops
        ])))
    ])
