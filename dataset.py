import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import imgaug.augmenters as iaa
import imutils
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage import exposure
from torch.utils.data import Dataset

import common

UP_CUT, BOTTOM_CUT = 10, 10
LEFT_CUT, RIGHT_CUT = 250, 250


class NormalizeHist:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = exposure.equalize_hist(img).astype(np.uint8)
        return img


class ElasticTransform:
    def __init__(self,
                 alpha: float,
                 sigma: float,
                 random_state: Optional[int] = None):
        self.random_state = random_state
        self.alpha = alpha
        self.sigma = sigma
        self.rng = np.random.RandomState(self.random_state)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        shape = img.shape
        dx = gaussian_filter((self.rng.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((self.rng.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distored_image = map_coordinates(img, indices, order=1, mode='reflect')
        return distored_image.reshape(img.shape)


class Denoising:
    def __init__(self, denoising_scale: int):
        self.denoising_scale = denoising_scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = cv2.fastNlMeansDenoising(img, h=self.denoising_scale)
        return img


class ImgaugWrapper:
    def __init__(self, augmenter: iaa.Augmenter):
        self.augmenter = augmenter

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self.augmenter.augment_image(img)
        return img


class ToBGR:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class UsgDataset(Dataset):
    def __init__(self,
                 paths: Union[List[Path], Tuple[Path]],
                 is_train_or_valid: bool,
                 has_crops: bool,
                 transforms: Optional[Callable] = None):
        self.paths = paths
        self.is_train_or_valid = is_train_or_valid
        self.transforms = transforms
        self.has_crops = has_crops

    def __getitem__(self, index):
        a_path = self.paths[index]
        names = ["lower.png", "radial_polar_area.png", "circle.png"]
        stack = []
        for name in names:
            img = cv2.imread((a_path / name).as_posix(), cv2.IMREAD_GRAYSCALE)
            if name == "lower.png":
                img = img[
                      UP_CUT + 110:img.shape[0] - BOTTOM_CUT,
                      LEFT_CUT:img.shape[1] - RIGHT_CUT
                      ]

            if name == "lower.png":
                img = imutils.resize(img, height=144, inter=cv2.INTER_LANCZOS4)

            if name == "circle.png":
                img = imutils.resize(img, height=144, width=144, inter=cv2.INTER_LANCZOS4)

            if name == "radial_polar_area.png":
                if img.shape[0] < img.shape[1] and img.shape[0] < 144:
                    img = imutils.resize(img, height=144, inter=cv2.INTER_LANCZOS4)
                elif img.shape[0] >= img.shape[1] and img.shape[1] < 144:
                    img = imutils.resize(img, width=144, inter=cv2.INTER_LANCZOS4)

            if self.transforms is not None:
                img = self.transforms(img)

            stack.append(img)

        if self.has_crops:
            img = np.concatenate(stack, axis=1)
        else:
            img = np.concatenate(stack, axis=0)
        if self.is_train_or_valid:
            a_class = int(a_path.parent.name)
            regression_data = json.loads(
                (a_path / common.REGRESSION_DATA_FILE_NAME).read_text()
            )

            split_class = int(regression_data["mean"] > common.BEST_SPLITTING_THRESHOLD)

            regression_value = (
                regression_data["mean"] - common.MIN_MEAN_DECIMAL_VALUE
            ) / common.MAX_VALUE_AFTER_SHIFT
            regression_value = np.asarray(regression_value).astype(np.float32)

            return img, (a_class, regression_value, split_class)

        return img, (-1, 0, 0)

    def __len__(self):
        return len(self.paths)
