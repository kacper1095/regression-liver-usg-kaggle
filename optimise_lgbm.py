import argparse
import json
import multiprocessing as mp
import pickle as pkl
import time
from pathlib import Path
from typing import Tuple
from pprint import pprint

import hyperopt as hp
import lightgbm as lgb
import numpy as np
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


def objective(params: dict, data_folder: str):
    print(json.dumps(params, indent=4))
    data_folder = Path(data_folder)

    train_paths, x_train = pkl.loads(
        (data_folder / "embeddings" / "train.pkl").read_bytes())
    valid_paths, x_valid = pkl.loads(
        (data_folder / "embeddings" / "valid.pkl").read_bytes())

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
        "num_leaves": int(params["num_leaves"]),
        "learning_rate": params["learning_rate"],
        "random_state": 0xCAFFE,
        "reg_alpha": params["reg_alpha"],
        "reg_lambda": params["reg_lambda"],
        "n_estimators": int(params["n_estimators"]),
        "min_split_gain": params["min_split_gain"],
        "subsample": params["subsample"],
        "verbose": 1
    }

    num_round = 100
    kfolds = 10

    folder = StratifiedKFold(n_splits=10, random_state=0xCAFFE)
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

        valid_predictions[valid_indices] = bst.predict(
            cur_x_valid, num_iteration=bst.best_iteration
        )

    val_rmse = rmse_as_metric(valid_predictions, y_train)
    return {
        "loss": val_rmse,
        "status": hp.STATUS_OK,
        "eval_time": time.time()
    }


def optimise(data_folder: str, out_folder: str):
    out_folder = Path(out_folder)
    out_folder.mkdir()

    space = {
        "n_estimators": hp.hp.quniform("n_estimators", 500, 2000, 1),
        "num_leaves": hp.hp.quniform("num_leaves", 16, 128, 1),
        "learning_rate": hp.hp.uniform("learning_rate", 0.001, 0.08),
        "reg_alpha": hp.hp.uniform("reg_alpha", 0.5, 4),
        "reg_lambda": hp.hp.uniform("reg_lambda", 0.5, 4),
        "min_split_gain": hp.hp.uniform("min_split_gain", 0.6, 0.9),
        "subsample": hp.hp.uniform("subsample", 0.6, 0.95)
    }
    trials = hp.Trials()

    obj_lambd = lambda params: objective(params, data_folder)
    best_params = hp.fmin(
        obj_lambd,
        space=space,
        trials=trials,
        max_evals=200,
        algo=hp.tpe.suggest
    )

    (out_folder / "trials.pkl").write_bytes(
        pkl.dumps(trials, protocol=pkl.HIGHEST_PROTOCOL)
    )
    (out_folder / "best_params.pkl").write_text(
        json.dumps(best_params, indent=4)
    )

    print(best_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_folder",
        help="Folder with 'train' and 'test' folders prepared for the competition."
    )
    parser.add_argument(
        "out_folder",
        help="Folder for optimisation_results."
    )
    args = parser.parse_args()
    optimise(args.data_folder, args.out_folder)


if __name__ == '__main__':
    main()
