# -*- coding: utf-8 -*-
from typing import Tuple, NoReturn

import boto3
import numpy as np
from sklearn.model_selection import train_test_split

from ..params import SplittingParams


def download_data_from_s3(s3_bucket: str, s3_path: str, output: str) -> NoReturn:
    s3 = boto3.client("s3")
    s3.download_file(s3_bucket, s3_path, output)


def split_train_val_data(data_path: str, params: SplittingParams) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray],
                                                                           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cache = np.load(data_path)
    images = cache['images']
    labels = cache['labels']
    paths = cache['paths']
    rgb_mean = cache['rgb_mean']
    rgb_std = cache['rgb_std']

    train_images, val_images, train_labels, val_labels, train_paths, val_paths = train_test_split(images, labels,
                                                                                                  paths,
                                                                                                  test_size=params.val_size,
                                                                                                  random_state=params.random_state)
    train_data = train_images, train_labels, train_paths
    val_data = val_images, val_labels, val_paths
    return train_data, val_data

