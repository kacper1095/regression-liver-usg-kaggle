import argparse
import multiprocessing as mp
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skorch import NeuralNet

import common
from common import get_timestamp, get_train_test_split_from_paths
from dataset import UsgDataset
from losses import MixedLoss
from metrics import fscore_as_metric, rmse_as_metric
from model import PretrainedModel
from train import batch_size
from transformers import get_test_transformers
from utils import get_true_values_from_paths, restore_real_prediction_values

torch.multiprocessing.set_sharing_strategy('file_system')


def generate_submission(data_folder: str, weights_path: str):
    weights_path = Path(weights_path)
    assert weights_path.exists()

    data_paths = list(
        chain(
            (Path(data_folder) / "train" / "0").glob("*"),
            (Path(data_folder) / "train" / "1").glob("*")
        )
    )
    data_paths = list(sorted(data_paths, key=lambda x: int(x.name)))

    classes = [int(path.parent.name) for path in data_paths]
    train_paths, valid_paths = get_train_test_split_from_paths(data_paths, classes)
    test_data_paths = list(
        (Path(data_folder) / "test").glob("*")
    )

    valid_dataset = UsgDataset(valid_paths,
                               True,
                               transforms=get_test_transformers(),
                               has_crops=True)

    net = NeuralNet(
        PretrainedModel,
        criterion=MixedLoss,
        module__extract_intermediate_values=False,
        module__n_dropout_runs=100,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=mp.cpu_count(),
        iterator_valid__batch_size=batch_size,
        device="cuda",
    )
    net.initialize()
    net.load_params(f_params=weights_path.as_posix())

    test_dataset = UsgDataset(test_data_paths,
                              is_train_or_valid=False,
                              transforms=get_test_transformers(),
                              has_crops=True)

    valid_predictions = net.forward(valid_dataset)
    valid_classification_predictions = F.softmax(
        valid_predictions[common.CLASSIFICATION_INDEX], dim=-1
    ).detach().cpu().numpy().mean(axis=1)
    valid_regression_predictions = restore_real_prediction_values(
        valid_predictions[common.REGRESSION_INDEX].detach().cpu().numpy()
    ).mean(axis=1)

    valid_classification_trues = np.asarray([int(path.parent.name) for path in valid_paths])
    valid_regression_trues = get_true_values_from_paths(valid_paths)

    val_acc = fscore_as_metric(valid_classification_predictions,
                               valid_classification_trues)
    val_rmse = rmse_as_metric(valid_regression_predictions, valid_regression_trues)

    test_predictions = net.forward(test_dataset)

    test_classification_predictions = F.softmax(
        test_predictions[common.CLASSIFICATION_INDEX], dim=-1
    ).detach().cpu().numpy().mean(axis=1)

    test_regression_predictions = restore_real_prediction_values(
        test_predictions[common.REGRESSION_INDEX].detach().cpu().numpy()
    ).mean(axis=1)

    ids = [path.name for path in test_data_paths]
    classes = np.argmax(test_classification_predictions, axis=1)

    timestamp = get_timestamp()

    print("Generating classification submission ... {:.4f}".format(val_acc))
    frame = pd.DataFrame(data={"id": ids, "label": classes})
    frame["id"] = frame["id"].astype(np.int32)
    frame = frame.sort_values(by=["id"])
    frame.to_csv(
        f"submissions/{timestamp}_{'%.4f' % val_acc}"
        f"_classification_submission.csv",
        index=False)

    print("Generating regression submission ... {:.4f}".format(val_rmse))
    frame = pd.DataFrame(data={"Id": ids,
                               "Predicted": test_regression_predictions})
    frame["Id"] = frame["Id"].astype(np.int)
    frame = frame.sort_values(by=["Id"])
    frame.to_csv(
        f"submissions/{timestamp}_{'%.4f' % val_rmse}"
        f"_regression_submission.csv",
        index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_folder",
        help="Folder with 'train' and 'test' folders prepared for the competition."
    )
    parser.add_argument(
        "model_folder",
        help="Folder with model to use for prediction."
    )

    args = parser.parse_args()
    generate_submission(args.data_folder, args.model_folder)


if __name__ == '__main__':
    main()
