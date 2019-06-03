import argparse
import json
import multiprocessing as mp
import pickle as pkl
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import common
from metrics import *


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


def train(data_folder: str, out_model: str):
    out_model = Path(out_model)
    out_model.mkdir()

    data_folder = Path(data_folder)

    train_paths, x_train = pkl.loads(
        (data_folder / "embeddings" / "train.pkl").read_bytes())
    valid_paths, x_valid = pkl.loads(
        (data_folder / "embeddings" / "valid.pkl").read_bytes())
    test_paths, x_test = pkl.loads((data_folder / "embeddings" / "test.pkl").read_bytes())

    y_train, cls_y_train = get_y_and_classes_from_ids(data_folder, train_paths)
    y_valid, cls_y_valid = get_y_and_classes_from_ids(data_folder, valid_paths)

    x_train = np.concatenate([x_train, x_valid], axis=0)
    y_train = np.concatenate([y_train, y_valid], axis=0)
    cls_y_train = np.concatenate([cls_y_train, cls_y_valid], axis=0)

    params = {
        "boosting_type": "gbdt",
        "max_depth": -1,
        "objective": "rmse",
        "metric": "rmse",
        "nthread": mp.cpu_count(),
        "num_leaves": 16,
        "learning_rate": 0.05,
        "random_state": 0xCAFFE,
        "reg_alpha": 1.2,
        "reg_lambda": 1.4,
        "n_estimators": 100,
        "min_split_gain": 0.5,
        "subsample": 0.85,
        "num_iterations": 1500
    }

    num_round = 100
    kfolds = 10

    folder = StratifiedKFold(n_splits=10, random_state=0xCAFFE)
    test_predictions = np.zeros((kfolds, len(test_paths)))
    valid_predictions = np.zeros_like(y_train)

    for i, (train_indices, valid_indices) in enumerate(folder.split(
            x_train, cls_y_train
    )):
        print("Fold: {} / {}".format(i + 1, kfolds))
        cur_x_train, cur_y_train = x_train[train_indices], y_train[train_indices]
        cur_x_valid, cur_y_valid = x_train[valid_indices], y_train[valid_indices]

        train_dataset = lgb.Dataset(cur_x_train, label=cur_y_train)
        valid_dataset = lgb.Dataset(cur_x_valid, label=cur_y_valid)

        bst = lgb.train(
            params,
            train_dataset,
            num_round,
            valid_sets=[valid_dataset],
            early_stopping_rounds=150
        )

        bst.save_model(
            (out_model / f"model_{i}.txt").as_posix(),
            num_iteration=bst.best_iteration
        )

        valid_predictions[valid_indices] = bst.predict(
            cur_x_valid, num_iteration=bst.best_iteration
        )
        test_predictions[i] = bst.predict(x_test, num_iteration=bst.best_iteration)

    print("Generating submission ...")
    val_rmse = rmse_as_metric(valid_predictions, y_train)
    test_ids = [int(Path(path).name) for path in test_paths]

    frame = pd.DataFrame(data={
        "Id": test_ids,
        "Predicted": test_predictions.mean(axis=0).round(1)
    })
    frame["Id"] = frame["Id"].astype(np.int)
    frame = frame.sort_values(by=["Id"])

    print("Generating submission ... {:.4f}".format(val_rmse))

    frame.to_csv(
        f"submissions/{common.get_timestamp()}_{'%.4f' % val_rmse}_submission.csv",
        index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_folder",
        help="Folder with 'train' and 'test' folders prepared for the competition."
    )
    parser.add_argument(
        "out_model",
        help="Output folder where weights and tensorboards will be saved."
    )
    args = parser.parse_args()
    train(args.data_folder, args.out_model)


if __name__ == '__main__':
    main()
