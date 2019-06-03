from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from skorch import NeuralNet
from torch.utils.data import Dataset

import common
from utils import restore_real_prediction_values

__all__ = [
    "acc", "acc_as_metric",
    "fscore_as_metric",
    "rmse", "rmse_as_metric",
    "fscore_for_classification",
    "fscore_for_split"
]


def acc_as_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (np.argmax(y_pred, axis=1) == y_true).mean().item()


def fscore_as_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return f1_score(y_true, np.argmax(y_pred, axis=1), average="macro")


def rmse_as_metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def acc(net: NeuralNet,
        ds: Optional[Dataset] = None,
        y: Optional[torch.Tensor] = None,
        y_pred: Optional[torch.Tensor] = None) -> float:
    if y_pred is None:
        y_pred = net.forward(ds)
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    return acc_as_metric(
        y_pred[common.CLASSIFICATION_INDEX].detach().cpu().numpy(),
        y
    )


def fscore(net: NeuralNet,
           prediction_index: int,
           ds: Optional[Dataset] = None,
           y: Optional[torch.Tensor] = None,
           y_pred: Optional[torch.Tensor] = None) -> float:
    if y_pred is None:
        y_pred = net.forward(ds)
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    return fscore_as_metric(
        y_pred[prediction_index].detach().cpu().numpy(),
        y
    )


def fscore_for_classification(net: NeuralNet,
                              ds: Optional[Dataset] = None,
                              y: Optional[torch.Tensor] = None,
                              y_pred: Optional[torch.Tensor] = None) -> float:
    return fscore(
        net, common.CLASSIFICATION_INDEX, ds, y, y_pred
    )


def fscore_for_split(net: NeuralNet,
                     ds: Optional[Dataset] = None,
                     y: Optional[torch.Tensor] = None,
                     y_pred: Optional[torch.Tensor] = None) -> float:
    return fscore(
        net, common.SPLIT_CLASSIFICATION_INDEX, ds, y, y_pred
    )


def rmse(net: NeuralNet,
         ds: Optional[Dataset] = None,
         y: Optional[torch.Tensor] = None,
         y_pred: Optional[torch.Tensor] = None) -> float:
    if y_pred is None:
        y_pred = net.forward(ds)
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    return rmse_as_metric(
        restore_real_prediction_values(
            y_pred[common.REGRESSION_INDEX].detach().cpu().numpy()
        ),
        restore_real_prediction_values(y)
    )
