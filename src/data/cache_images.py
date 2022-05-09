import os
import random
from collections import Counter
from pathlib import Path
from typing import NoReturn, List, Tuple

import click
import numpy as np
import yaml
from PIL import Image, ImageStat
from marshmallow import Schema
from tqdm.notebook import tqdm

from ..params.params import PreprocessingParams


def eliminate_rare(images: List, labels: List, paths: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rare_labels = [lbl for lbl, count in Counter(labels).items() if count < 2]
    rare_indexes = []  # Indexes of samples, where labels are rather rare
    if any(rare_labels):
        for j, lbl in tqdm(enumerate(labels)):
            if lbl in rare_labels:
                rare_indexes.append(j)
    images_np = np.stack([_ for (i, _) in enumerate(images) if i not in rare_indexes])
    labels_np = np.stack([_ for (i, _) in enumerate(labels) if i not in rare_indexes])
    paths_np = np.stack([_ for (i, _) in enumerate(paths) if i not in rare_indexes])
    return images_np, labels_np, paths_np


def cache_images(preprocessing_params: PreprocessingParams) -> NoReturn:
    images_folder = preprocessing_params.input_data_path
    cache_file = preprocessing_params.cache_file_path
    processed_images_folder = preprocessing_params.processed_images_folder

    wh = (preprocessing_params.width, preprocessing_params.height)
    save_processed_images = preprocessing_params.save_processed_images  # It may be useful when you want to look at
    # processed (scaled) images.

    if not Path(images_folder).exists():
        raise RuntimeError(f"Folder {images_folder} doesn't exists.")

    data = []
    for folder, _, files in os.walk(images_folder):
        for file_name in files:
            rel_dir = os.path.relpath(folder, images_folder)
            label = tuple(rel_dir.split(os.sep))
            file_path = os.path.join(folder, file_name)
            if file_path.split('.')[-1].lower() in ['bmp', 'jpg', 'jpeg', 'png']:
                data.append((file_path, label))

    k = min(len(data), 30)  # There is no need to calculate mean and standard deviation over all samples, it's enough
    # to take 30, for instance.

    mean = []
    std = []
    for path, _ in random.choices(data, k=k):
        image_pil = Image.open(path)
        stat = ImageStat.Stat(image_pil)
        mean.append(stat.mean)
        std.append(stat.stddev)

    rgb_mean = np.mean(std, axis=0)
    rgb_std = np.mean(mean, axis=0)

    images = []
    labels = []
    paths = []
    for path, label in data:
        with Image.open(path) as image:
            blank_image = Image.new('RGB', wh, 'black')
            image.thumbnail(wh, Image.BICUBIC)
            blank_image.paste(image, (0, 0))  # add to left upper corner
            image_pil_np = np.asarray(blank_image)
            images.append((image_pil_np - rgb_mean) / rgb_std)  # Standardization
            labels.append(label)
            paths.append(path)
            if save_processed_images:
                blank_image.save(os.path.join(processed_images_folder, '_'.join(path.split(os.sep)[-3:])))

    labels = [str(lbl) for lbl in labels]
    images, labels, paths = eliminate_rare(images, labels, paths)
    assert images.shape[0] == labels.shape[0] == paths.shape[0]
    np.savez(cache_file, images=images, labels=labels, paths=paths, rgb_mean=rgb_mean, rgb_std=rgb_std)


@click.command(name="cache_image")
@click.argument("config_path")
def main(config_path: str):
    with open(config_path, "r") as input_stream:
        preprocessing_params = Schema(PreprocessingParams).load(yaml.safe_load(input_stream))
    cache_images(preprocessing_params)


if __name__ == "__main__":
    main()
