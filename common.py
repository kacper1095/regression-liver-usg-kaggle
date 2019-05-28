import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import imgaug.augmenters as iaa
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from skorch.net import NeuralNet
from torch.utils.data import Dataset
from torchvision.transforms import Compose, FiveCrop, Lambda, RandomCrop, Resize, \
    ToPILImage, ToTensor

from dataset import Denoising, ImgaugWrapper


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


def acc_as_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (np.argmax(y_pred, axis=1) == y_true).mean().item()


def acc(net: NeuralNet,
        ds: Optional[Dataset] = None,
        y: Optional[torch.Tensor] = None,
        y_pred: Optional[torch.Tensor] = None) -> float:
    if y_pred is None:
        y_pred = net.predict(ds)
    return acc_as_metric(y_pred, y)


def fscore_as_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return f1_score(y_true, np.argmax(y_pred, axis=1), average="weighted")


def fscore(net: NeuralNet,
           ds: Optional[Dataset] = None,
           y: Optional[torch.Tensor] = None,
           y_pred: Optional[torch.Tensor] = None) -> float:
    if y_pred is None:
        y_pred = net.predict(ds)
    return fscore_as_metric(y_pred, y)


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%HH_%MM_%dd_%mm_%Yy")


def get_train_test_split_from_paths(data_paths: List[Path], classes: List[int]) -> Tuple[
    List[Path], List[Path]]:
    train_paths, valid_paths = train_test_split(data_paths, test_size=0.3,
                                                stratify=classes, random_state=0)
    return train_paths, valid_paths
