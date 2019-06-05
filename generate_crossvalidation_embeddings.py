import argparse
import multiprocessing as mp
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from skorch import NeuralNet
from skorch.callbacks import ProgressBar

import common
from generate_embeddings import get_prediction_with_paths, save_data_to_path
from losses import MixedLoss
from model import PretrainedModel
from train import balance_paths_by_decimal_value, batch_size
from transformers import get_test_transformers_with_augmentations

torch.multiprocessing.set_sharing_strategy('file_system')
USE_CUDA = torch.cuda.is_available()


def generate_crossvalidation_embeddings(
        data_folder: str, weights_path: str, output_path: str
):
    weights_path = Path(weights_path)
    output_path = Path(output_path)

    data_paths = list(
        chain(
            (Path(data_folder) / "train" / "0").glob("*"),
            (Path(data_folder) / "train" / "1").glob("*")
        )
    )
    data_paths = np.asarray(list(sorted(data_paths, key=lambda x: int(x.name))))
    classes = [int(path.parent.name) for path in data_paths]
    test_paths = list(
        (Path(data_folder) / "test").glob("*")
    )

    folder = StratifiedKFold(n_splits=common.K_FOLDS,
                             random_state=common.RANDOM_STATE_SEED)

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

    print("Saving test embeddings ...")
    save_data_to_path(get_prediction_with_paths(test_paths, net),
                      output_path / "test.pkl")

    for i, (train_indices, valid_indices) in enumerate(folder.split(
            data_paths, classes
    )):
        print("Fold: {} / {}".format(i + 1, common.K_FOLDS))
        train_paths = data_paths[train_indices]
        valid_paths = data_paths[valid_indices]

        print("Saving train embeddings for fold {} ...".format(i + 1))
        save_data_to_path(
            get_prediction_with_paths(
                list(balance_paths_by_decimal_value(train_paths)),
                net,
                get_test_transformers_with_augmentations()
            ),
            output_path / "train_{}.pkl".format(i)
        )

        print("Saving valid embeddings for fold {} ...".format(i + 1))
        save_data_to_path(get_prediction_with_paths(valid_paths, net),
                          output_path / "valid_{}.pkl".format(i))


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
    generate_crossvalidation_embeddings(args.data_folder,
                                        args.model_folder,
                                        args.output_path)


if __name__ == '__main__':
    main()
