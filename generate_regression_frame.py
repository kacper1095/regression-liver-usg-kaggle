import argparse
import multiprocessing as mp
from itertools import chain
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skorch import NeuralNet
from skorch.callbacks import ProgressBar
from torchvision.transforms import Compose

import common
from common import get_train_test_split_from_paths
from dataset import UsgDataset
from losses import MixedLoss
from model import PretrainedModel
from train import batch_size
from transformers import get_test_transformers
from utils import get_true_values_from_paths, restore_real_prediction_values

torch.multiprocessing.set_sharing_strategy('file_system')


def to_numpy(data: torch.Tensor) -> np.ndarray:
    return data.detach().cpu().numpy()


def generate_dataframe(data_folder: str, weights_path: str, output_path: str):
    weights_path = Path(weights_path)
    output_path = Path(output_path)

    data_paths = list(
        chain(
            (Path(data_folder) / "train" / "0").glob("*"),
            (Path(data_folder) / "train" / "1").glob("*")
        )
    )
    data_paths = list(sorted(data_paths, key=lambda x: int(x.name)))

    classes = [int(path.parent.name) for path in data_paths]

    _, valid_paths = get_train_test_split_from_paths(data_paths, classes)

    net = NeuralNet(
        PretrainedModel,
        criterion=MixedLoss,
        module__extract_intermediate_values=True,
        module__n_dropout_runs=common.N_DROPOUT_INFERENCES,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=mp.cpu_count(),
        iterator_valid__batch_size=batch_size,
        device="cuda",
        callbacks=[ProgressBar()]
    )
    net.initialize()
    net.load_params(f_params=weights_path.as_posix())

    print("Producing DataFrame valid embeddings ...")
    frame = get_frame_with_predictions(valid_paths, net)
    frame.to_csv(output_path / "predictions.csv", index=False)


def get_frame_with_predictions(
        paths: Union[List[Path], Tuple[Path]],
        net: NeuralNet,
        transformers: Compose = get_test_transformers()
) -> pd.DataFrame:
    dataset = UsgDataset(
        paths, is_train_or_valid=False,
        transforms=transformers,
        has_crops=True
    )

    true_regression = get_true_values_from_paths(paths)
    true_classification = np.asarray([int(path.parent.name) for path in paths])

    predictions = list(net.forward(dataset))

    classes_predictions = predictions[0]
    regression_predictions = predictions[1]

    classes_predictions = F.softmax(classes_predictions, dim=-1)
    regression_predictions = restore_real_prediction_values(regression_predictions)

    stats = []

    for datum in [
        classes_predictions,
        regression_predictions
    ]:
        for a_fun in [torch.mean, torch.std]:
            stats.append(a_fun(datum, dim=1))
    paths = np.asarray([path.as_posix() for path in paths])

    cls_mean_predictions = to_numpy(torch.mean(classes_predictions, dim=1))
    cls_std_predictions = to_numpy(torch.std(classes_predictions, dim=1))

    reg_mean_predictions = to_numpy(torch.mean(regression_predictions, dim=1))
    reg_std_predictions = to_numpy(torch.std(regression_predictions, dim=1))

    data = {
        "path": paths,

        "reg_true": true_regression,
        "cls_true": true_classification,

        "reg_mean_pred": reg_mean_predictions,
        "reg_std_pred": reg_std_predictions,

        "f0_mean_pred": cls_mean_predictions[..., 0],
        "f4_mean_pred": cls_mean_predictions[..., 1],

        "f0_std_pred": cls_std_predictions[..., 0],
        "f4_std_pred": cls_std_predictions[..., 1]
    }
    frame = pd.DataFrame(data=data)
    return frame


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
    parser.add_argument(
        "output_path",
        help="Output path for pickled data: train, valid, test"
    )

    args = parser.parse_args()
    generate_dataframe(args.data_folder, args.model_folder, args.output_path)


if __name__ == '__main__':
    main()
