import argparse
import multiprocessing as mp
import pickle as pkl
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import common
from metrics import *
from utils import get_y_and_classes_from_ids


def train(data_folder: str, out_model: str):
    out_model = Path(out_model)
    out_model.mkdir()

    data_folder = Path(data_folder)
    test_paths, x_test = pkl.loads((data_folder
                                    / "cv_embeddings"
                                    / "test.pkl").read_bytes())

    params = {
        "boosting_type": "gbdt",
        "max_depth": -1,
        "objective": "rmse",
        "metric": "rmse",
        "nthread": mp.cpu_count(),
        "num_leaves": 26,
        "learning_rate": 0.05,
        "random_state": 0xCAFFE,
        "reg_alpha": 1.2,
        "reg_lambda": 1.4,
        "n_estimators": 1500,
        "min_split_gain": 0.8,
        "subsample": 0.85,
    }

    num_round = 100
    test_predictions = []
    predictions_valid = []
    trues_valid = []

    for i in range(common.K_FOLDS):
        print("Fold: {} / {}".format(i + 1, common.K_FOLDS))

        train_paths, x_train = pkl.loads((data_folder
                                          / "cv_embeddings"
                                          / "train_{}.pkl".format(i)).read_bytes())

        valid_paths, x_valid = pkl.loads((data_folder
                                          / "cv_embeddings"
                                          / "valid_{}.pkl".format(i)).read_bytes())

        y_train, _ = get_y_and_classes_from_ids(data_folder, train_paths)
        y_valid, _ = get_y_and_classes_from_ids(data_folder, valid_paths)

        train_dataset = lgb.Dataset(x_train, label=y_train)
        valid_dataset = lgb.Dataset(x_valid, label=y_valid)

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

        predictions_valid.append(bst.predict(x_valid, num_iteration=bst.best_iteration))
        trues_valid.append(y_valid)

        test_predictions.append(bst.predict(x_test, num_iteration=bst.best_iteration))

    test_predictions = np.asarray(test_predictions)

    predictions_valid = np.concatenate(predictions_valid, axis=0)
    trues_valid = np.concatenate(trues_valid, axis=0)

    print("Generating submission ...")
    val_rmse = rmse_as_metric(predictions_valid, trues_valid)
    test_ids = [int(Path(path).name) for path in test_paths]

    frame = pd.DataFrame(data={
        "Id": test_ids,
        "Predicted": test_predictions.mean(axis=0)
    })
    frame["Id"] = frame["Id"].astype(np.int)
    frame = frame.sort_values(by=["Id"])

    print("Generating submission ... {:.4f}".format(val_rmse))

    frame.to_csv(
        (out_model
         / f"{common.get_timestamp()}_{'%.4f' % val_rmse}_submission.csv"),
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
