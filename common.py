import datetime
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

MIN_MEAN_DECIMAL_VALUE = 3
MAX_VALUE_AFTER_SHIFT = 61
REGRESSION_DATA_FILE_NAME = "regression_ground_truth.json"

CLASSIFICATION_INDEX = 0
REGRESSION_INDEX = 1


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%HH_%MM_%dd_%mm_%Yy")


def get_train_test_split_from_paths(data_paths: List[Path], classes: List[int]) -> Tuple[
    List[Path], List[Path]]:
    train_paths, valid_paths = train_test_split(data_paths, test_size=0.3,
                                                stratify=classes, random_state=0)
    return train_paths, valid_paths
