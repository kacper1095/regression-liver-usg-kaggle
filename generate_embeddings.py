import argparse
import multiprocessing as mp
import pickle as pkl
from itertools import chain
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from skorch import NeuralNet
from skorch.callbacks import ProgressBar

from common import get_train_test_split_from_paths
from dataset import UsgDataset
from losses import MixedLoss
from model import PretrainedModel
from train import batch_size
from transformers import get_test_transformers
from utils import restore_real_prediction_values

torch.multiprocessing.set_sharing_strategy('file_system')


def generate_embeddings(data_folder: str, weights_path: str, output_path: str):
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

    train_paths, valid_paths = get_train_test_split_from_paths(data_paths, classes)
    test_paths = list(
        (Path(data_folder) / "test").glob("*")
    )

    net = NeuralNet(
        PretrainedModel,
        criterion=MixedLoss,
        module__extract_intermediate_values=True,
        module__n_dropout_runs=100,
        iterator_valid__shuffle=False,
        iterator_valid__num_workers=mp.cpu_count(),
        iterator_valid__batch_size=batch_size,
        device="cuda",
        callbacks=[ProgressBar()]
    )
    net.initialize()
    net.load_params(f_params=weights_path.as_posix())

    print("Saving train embeddings ...")
    save_data_to_path(get_prediction_with_paths(train_paths, net),
                      output_path / "train.pkl")

    print("Saving valid embeddings ...")
    save_data_to_path(get_prediction_with_paths(valid_paths, net),
                      output_path / "valid.pkl")

    print("Saving test embeddings ...")
    save_data_to_path(get_prediction_with_paths(test_paths, net),
                      output_path / "test.pkl")


def get_prediction_with_paths(
        paths: Union[List[Path], Tuple[Path]],
        net: NeuralNet,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = UsgDataset(
        paths, is_train_or_valid=False,
        transforms=get_test_transformers(),
        has_crops=True
    )

    predictions = list(net.forward(dataset))

    classes_predictions = predictions[0]
    regression_predictions = predictions[1]
    split_predictions = predictions[2]

    # because all embeddings are the same
    pooled_features_predictions = predictions[3][:, 0]

    classes_predictions = F.softmax(classes_predictions, dim=-1)
    split_predictions = F.softmax(split_predictions, dim=-1)
    regression_predictions = restore_real_prediction_values(regression_predictions)
    regression_predictions = regression_predictions.unsqueeze(-1)

    stats = []

    for datum in [
        classes_predictions,
        split_predictions,
        regression_predictions
    ]:
        for a_fun in [torch.mean, torch.var, torch.std]:
            stats.append(a_fun(datum, dim=1))

    final_predictions = torch.cat((torch.cat(tuple(stats), dim=-1),
                                   pooled_features_predictions),
                                  dim=-1).detach().cpu().numpy()

    paths = np.asarray([path.as_posix() for path in paths])

    return paths, final_predictions


def save_data_to_path(data: Any, path: Path):
    path.write_bytes(pkl.dumps(data, protocol=pkl.HIGHEST_PROTOCOL))


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
    generate_embeddings(args.data_folder, args.model_folder, args.output_path)


if __name__ == '__main__':
    main()
