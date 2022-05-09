import logging
import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .net import Network, TripletLoss, CachedDataset
from ..params import TrainingParams, TestParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_net_model(train_data: Tuple[np.ndarray, np.ndarray, np.ndarray], val_data: Tuple[np.ndarray, np.ndarray, np.ndarray], train_params: TrainingParams) -> Tuple[Network, Optimizer]:
    if train_params.model_net_type == "TripletLoss":
        model = Network(emb_dim=train_params.embedding_size)
    else:
        raise NotImplementedError(f"Net type {train_params.model_net_type} is not supported yet.")

    logger.info("Training the net is started.")
    torch.manual_seed(train_params.seed)
    np.random.seed(train_params.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_images, train_labels, train_paths = train_data
    train_ds = CachedDataset(train_images, train_labels, train_paths)
    train_loader = DataLoader(train_ds, batch_size=train_params.batch_size, shuffle=True)

    val_images, val_labels, val_paths = val_data
    test_ds = CachedDataset(val_images, val_labels, val_paths)
    test_loader = DataLoader(test_ds, batch_size=train_params.batch_size, shuffle=False, num_workers=4)

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)

    model.apply(init_weights)
    model = torch.jit.script(model).to(device)
    criterion = torch.jit.script(TripletLoss())
    optimizer = optim.Adam(model.parameters(), lr=train_params.lr)

    for epoch in tqdm(range(train_params.epochs), desc="Epochs"):
        running_loss = []
        model.train()
        for step, (anchor_img, positive_img, negative_img, anchor_label, path) in enumerate(
                tqdm(train_loader, desc="Training", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            optimizer.zero_grad()
            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.cpu().detach().numpy() * anchor_img.shape[0] / len(train_ds))
        if (epoch + 1) % 10 == 0:
            val_running_loss = []
            model.eval()
            with torch.no_grad():
                for step, (anchor_img, positive_img, negative_img, anchor_label, path) in enumerate(
                        tqdm(test_loader, desc="Evaluating", leave=False)):
                    anchor_img = anchor_img.to(device)
                    positive_img = positive_img.to(device)
                    negative_img = negative_img.to(device)

                    anchor_out = model(anchor_img)
                    positive_out = model(positive_img)
                    negative_out = model(negative_img)

                    loss = criterion(anchor_out, positive_out, negative_out)

                    val_running_loss.append(loss.cpu().detach().numpy() * anchor_img.shape[0] / len(test_ds))

            print(f"\rEpoch: {epoch + 1:3d}/{train_params.epochs:3d} - Train Loss: {np.sum(running_loss):.4f}, Val Loss: {np.sum(val_running_loss):.4f}", end='')
    logger.info("Training the net is finished.")
    return model, optimizer


def predict_net_model(model: Network, test_images: np.ndarray, test_params: TestParams) -> np.ndarray:
    ds = CachedDataset(test_images, None, None, return_paths=False, mode="test")
    loader = DataLoader(ds, batch_size=test_params.batch_size, shuffle=False)

    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for img, _, _, _, _ in tqdm(loader, desc="Predicting..."):
            results.append(model(img.to(device)).cpu().numpy())

    embeddings = np.concatenate(results)

    return embeddings


def serialize_net_model(model: Network, optimizer: Optimizer, output: str) -> str:
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, output)
    return output

