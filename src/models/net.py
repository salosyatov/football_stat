from random import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, images, labels, paths, return_paths=False, mode="train"):
        self.return_paths = return_paths

        self.images = torch.from_numpy(images.transpose(0, 3, 1, 2)).float()
        self.labels = labels
        self.paths = paths
        self.index = np.arange(len(images))
        self.mode = mode

        assert len(self.images) == len(self.labels) == len(self.index) == len(self.paths)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        anchor_label = self.labels[item]
        anchor_img = self.images[item]

        if self.mode == "train" or self.mode == "val":
            positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            if positive_list.any():
                positive_item = random.choice(positive_list)
                positive_img = self.images[positive_item]
            else:
                raise Exception('positive_list cannot be empty')

            negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]
            if negative_list.any():
                negative_item = random.choice(negative_list)
                negative_img = self.images[negative_item]
            else:
                raise Exception('negative_list cannot be empty')
        elif self.mode == "test":
            positive_img, negative_img, anchor_label = None, None, None

        returned_path = self.paths[item] if self.return_paths else np.nan
        return anchor_img, positive_img, negative_img, anchor_label, returned_path


class TripletLoss(nn.Module):
    def __init__(self, margin=0.25):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class Network(nn.Module):
    def __init__(self, emb_dim=128):
        super(Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16 * 2, 5, padding=2),  # (H, W)
            nn.PReLU(),
            nn.MaxPool2d(2),  # (H // 2, W // 2)
            nn.Dropout(0.3),

            nn.Conv2d(16 * 2, 32 * 2, 5, padding=2),  # (H // 2, W // 2)
            nn.PReLU(),
            nn.MaxPool2d(2),  # (H // 4, W // 4)
            nn.Dropout(0.3),

            nn.Conv2d(32 * 2, 64 * 2, 3, padding=1),  # (H // 4, W // 4)
            nn.PReLU(),
            nn.MaxPool2d(2),  # (H // 8, W // 8)
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * (HW[0] // 8) * (HW[1] // 8), 256),
            nn.PReLU(),
            nn.Linear(256, emb_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc(x)
        # x = nn.functional.normalize(x, p=1.0)
        return x