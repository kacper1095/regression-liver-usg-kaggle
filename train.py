import argparse
import multiprocessing as mp
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from skorch.callbacks import BatchScoring, Checkpoint, EpochScoring, LRScheduler, \
    ProgressBar
from skorch.helper import predefined_split

import common
from dataset import UsgDataset
from losses import *
from metrics import *
from model import PretrainedModel
from transformers import *
from utils import *

# Needed it because of in `DataLoader` for validation set
# RuntimeError: received 0 items of ancdata
# https://github.com/pytorch/pytorch/issues/973#issuecomment-426559250
torch.multiprocessing.set_sharing_strategy('file_system')
batch_size = 16


def train(data_folder: str, out_model: str):
    out_model = Path(out_model)
    out_model.mkdir()

    data_paths = list(
        chain(
            (Path(data_folder) / "train" / "0").glob("*"),
            (Path(data_folder) / "train" / "1").glob("*")
        )
    )
    data_paths = list(sorted(data_paths, key=lambda x: int(x.name)))

    classes = [int(path.parent.name) for path in data_paths]
    train_paths, valid_paths = common.get_train_test_split_from_paths(data_paths, classes)

    train_dataset = UsgDataset(train_paths, True,
                               transforms=get_train_transformers(),
                               has_crops=False)
    valid_dataset = UsgDataset(valid_paths, True,
                               transforms=get_test_transformers(),
                               has_crops=True)
    net = NeuralNet(
        PretrainedModel,
        criterion=MixedLoss,
        batch_size=batch_size,
        max_epochs=100,
        optimizer=optim.Adam,
        lr=0.0001,
        iterator_train__shuffle=True,
        iterator_train__num_workers=mp.cpu_count(),
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=mp.cpu_count(),
        train_split=predefined_split(valid_dataset),
        device="cuda",
        callbacks=[
            Checkpoint(
                f_params=(out_model / "params.pt").as_posix(),
                f_optimizer=(out_model / "optim.pt").as_posix(),
                f_history=(out_model / "history.pt").as_posix()
            ),

            EpochScoring(fscore,
                         name="val_fscore",
                         lower_is_better=False,
                         on_train=False,
                         target_extractor=lambda x: x[0]),
            EpochScoring(rmse,
                         name="val_rmse",
                         lower_is_better=True,
                         on_train=False,
                         target_extractor=lambda x: x[1]),

            BatchScoring(fscore,
                         name="train_fscore",
                         lower_is_better=False,
                         on_train=True,
                         target_extractor=lambda x: x[0]),
            BatchScoring(rmse,
                         name="train_rmse",
                         lower_is_better=True,
                         on_train=True,
                         target_extractor=lambda x: x[1]),

            ProgressBar(postfix_keys=[
                "train_loss",
                "train_fscore",
                "train_rmse"
            ]),
            LRScheduler(
                policy="ReduceLROnPlateau",
                monitor="valid_loss",
                factor=0.91,
                patience=3,
            ),
        ],
        warm_start=True
    )

    net.fit(train_dataset)

    print("Generating submission ...")
    net = NeuralNet(
        PretrainedModel,
        criterion=nn.CrossEntropyLoss,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=mp.cpu_count(),
        iterator_valid__batch_size=batch_size,
        device="cuda",
    )
    net.initialize()
    net.load_params(f_params=(out_model / "params.pt").as_posix())

    test_data_paths = list(
        (Path(data_folder) / "test").glob("*")
    )
    test_dataset = UsgDataset(
        test_data_paths, is_train_or_valid=False,
        transforms=get_test_transformers(),
        has_crops=True
    )

    valid_predictions = net.forward(valid_dataset)
    valid_predictions = restore_real_prediction_values(
        valid_predictions[common.REGRESSION_INDEX].detach().cpu().numpy()
    )

    valid_trues = get_true_values_from_paths(valid_paths)
    val_rmse = rmse_as_metric(valid_predictions, valid_trues)

    predictions = net.forward(test_dataset)
    predictions = restore_real_prediction_values(
        predictions[common.REGRESSION_INDEX].detach().cpu().numpy()
    )

    ids = [path.name for path in test_data_paths]
    frame = pd.DataFrame(data={"Id": ids, "Predicted": predictions})
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
