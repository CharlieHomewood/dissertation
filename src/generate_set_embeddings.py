import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils.spatial_embeddings import (
    freeze_frame_to_array,
    FreezeFramePointAutoencoder,
)


model_path = "models/freeze_frame_point_autoencoder.pt"
input_dir = "data/actions"
max_players = 22
batch_size= 512


class FreezeFrameEmbeddingDataset(Dataset):
    """
    Dataset for embedding generation.

    Returns the original row index so embeddings can be written back
    to the correct dataframe row.
    """

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        freeze_frame_array = freeze_frame_to_array(
            row["freeze_frame"],
            flip=(row["team_id"] != row["home_team_id"]),
        )

        return idx, freeze_frame_array


def collate_embedding_rows(batch):
    indices, freeze_frames = zip(*batch)

    batch_size = len(freeze_frames)
    feature_dim = 5

    max_players = max(ff.shape[0] for ff in freeze_frames)
    max_players = max(max_players, 1)

    x = torch.zeros(
        batch_size,
        max_players,
        feature_dim,
        dtype=torch.float32,
    )

    padding_mask = torch.ones(
        batch_size,
        max_players,
        dtype=torch.bool,
    )

    for i, freeze_frame in enumerate(freeze_frames):
        n_players = freeze_frame.shape[0]

        if n_players > 0:
            x[i, :n_players] = torch.from_numpy(freeze_frame)
            padding_mask[i, :n_players] = False

    indices = torch.tensor(indices, dtype=torch.long)

    return indices, x, padding_mask


def load_trained_model(model_path: str, device: str):
    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    params = checkpoint["best_params"]

    model = FreezeFramePointAutoencoder(
        embedding_dim=params["embedding_dim"],
        max_players=max_players,
        hidden_dim=params["hidden_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, params["embedding_dim"]


def remove_existing_embedding_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        col for col in df.columns
        if col.startswith("spatial_embedding_")
        or col == "has_freeze_frame"
    ]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def append_embeddings(
    df: pd.DataFrame,
    model: FreezeFramePointAutoencoder,
    embedding_dim: int,
    device: str,
    batch_size: int = batch_size,
) -> pd.DataFrame:
    dataset = FreezeFrameEmbeddingDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_embedding_rows,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    embedding_array = np.zeros(
        (len(df), embedding_dim),
        dtype=np.float32,
    )

    has_freeze_frame = np.zeros(
        len(df),
        dtype=np.int8,
    )

    with torch.no_grad():
        for indices, x, padding_mask in tqdm(
            loader,
            desc="Generating embeddings",
        ):
            valid_rows = (~padding_mask).any(dim=1)

            batch_indices = indices.numpy()
            has_freeze_frame[batch_indices[valid_rows.numpy()]] = 1

            if not valid_rows.any():
                continue

            valid_indices = indices[valid_rows]

            x_valid = x[valid_rows].to(
                device,
                non_blocking=True,
            )

            padding_mask_valid = padding_mask[valid_rows].to(
                device,
                non_blocking=True,
            )

            embeddings = model.encoder(
                x_valid,
                padding_mask_valid,
            )

            embedding_array[valid_indices.numpy()] = (
                embeddings.cpu().numpy()
            )

    embedding_cols = [
        f"spatial_embedding_{i:02d}"
        for i in range(embedding_dim)
    ]

    embedding_df = pd.DataFrame(
        embedding_array,
        columns=embedding_cols,
        index=df.index,
    )

    df_with_embeddings = pd.concat(
        [df, embedding_df],
        axis=1,
    )

    df_with_embeddings["has_freeze_frame"] = has_freeze_frame

    return df_with_embeddings


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, embedding_dim = load_trained_model(
        model_path,
        device,
    )

    action_paths = [
        Path(input_dir) / file
        for file in os.listdir(input_dir)
        if file.endswith(".parquet")
    ]

    print(f"Found {len(action_paths)} parquet files.")

    for action_path in action_paths:
        print(f"\nReading {action_path}")

        df = pd.read_parquet(action_path)

        df = remove_existing_embedding_columns(df)

        df_with_embeddings = append_embeddings(
            df=df,
            model=model,
            embedding_dim=embedding_dim,
            device=device,
            batch_size=batch_size,
        )

        df_with_embeddings.to_parquet(
            action_path,
            index=False,
        )

        print(f"Overwritten: {action_path}")
        print(f"Shape: {df_with_embeddings.shape}")


if __name__ == "__main__":
    main()