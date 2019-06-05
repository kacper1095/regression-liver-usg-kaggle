import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

import common

__all__ = [
    "get_true_values_from_paths",
    "get_full_number_from_all_predictions",
    "get_y_and_classes_from_ids",
    "restore_real_prediction_values"
]


def get_full_number_from_all_predictions(
        decimal_predictions: np.ndarray,
        factorial_predictions: np.ndarray
) -> np.ndarray:
    decimal_predictions = decimal_predictions.astype(np.float32)
    factorial_predictions = factorial_predictions.astype(np.float32)

    if len(decimal_predictions.shape) == 2:
        decimal_predictions = decimal_predictions.argmax(axis=1)
        factorial_predictions = factorial_predictions.argmax(axis=1)

    decimal_predictions = decimal_predictions + common.MIN_MEAN_DECIMAL_VALUE
    factorial_predictions = factorial_predictions / 10
    return decimal_predictions + factorial_predictions


def restore_real_prediction_values(predictions: np.ndarray) -> np.ndarray:
    return (predictions.squeeze() * common.MAX_VALUE_AFTER_SHIFT) \
           + common.MIN_MEAN_DECIMAL_VALUE


def get_true_values_from_paths(paths: Iterable[Path]) -> np.ndarray:
    values = []
    for path in paths:
        datum = json.loads((path / common.REGRESSION_DATA_FILE_NAME).read_text())
        values.append(datum["mean"])
    return np.asarray(values)


def get_y_and_classes_from_ids(
        original_samples_path: Path,
        file_paths: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    gt = np.zeros_like(file_paths, dtype=np.float32)
    classes = np.zeros_like(file_paths, dtype=np.int)

    for i, a_path in enumerate(file_paths):
        a_path = Path(a_path)
        datum = json.loads(
            (
                    original_samples_path
                    / a_path.parent.parent.name
                    / a_path.parent.name
                    / a_path.name
                    / common.REGRESSION_DATA_FILE_NAME
            ).read_text()
        )["mean"]
        gt[i] = datum
        classes[i] = int(a_path.parent.name)
    return gt, classes
