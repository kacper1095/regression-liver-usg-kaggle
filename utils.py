import json
from pathlib import Path
from typing import Iterable, Tuple

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import common

__all__ = [
    "get_true_values_from_paths",
    "get_full_number_from_all_predictions",
    "get_y_and_classes_from_ids",
    "restore_real_prediction_values",
    "plot_importances"
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


def plot_importances(
        clf: lgbm.Booster,
        show: bool = True,
        limit: int = 100
) -> Tuple[np.ndarray, pd.DataFrame]:
    columns = [
        nam + "_" + op
        for nam in ["cls", "split", "reg"]
        for op in ["mean", "var", "std"]
    ]
    columns += [f"feat_{i}" for i in list(range(1039 - len(columns)))]
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(), columns)),
                               columns=['Value', 'Feature'])

    feature_imp = feature_imp.sort_values(by="Value", ascending=False)

    fig = plt.figure(figsize=(20, 20))
    sns.barplot(x="Value", y="Feature", data=feature_imp.iloc[:limit])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8) \
        .reshape((int(height), int(width), 3))
    if show:
        plt.show()
    return image, feature_imp
