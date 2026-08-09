import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def freeze_frame_to_array(
    freeze_frame,
    flip: bool = False,
) -> np.ndarray:

    if isinstance(freeze_frame, str):
        try:
            freeze_frame = json.loads(freeze_frame)
        except json.JSONDecodeError:
            return np.zeros((0, 5), dtype=np.float32)

    if not isinstance(freeze_frame, list):
        return np.zeros((0, 5), dtype=np.float32)

    rows = []

    for p in freeze_frame:
        loc = p.get("location")

        if not isinstance(loc, list) or len(loc) != 2:
            continue

        sb_x, sb_y = loc

        SB_LENGTH = 120
        SB_WIDTH = 80

        SPADL_LENGTH = 105
        SPADL_WIDTH = 68

        x = (sb_x - 0.05) / SB_LENGTH * SPADL_LENGTH
        y = SPADL_WIDTH - (sb_y - 0.05) / SB_WIDTH * SPADL_WIDTH

        if flip:
            x = SPADL_LENGTH - x
            y = SPADL_WIDTH - y

        rows.append([
            x,
            y,
            float(p.get("teammate") is True),
            float(p.get("actor") is True),
            float(p.get("keeper") is True),
        ])

    return np.array(rows, dtype=np.float32) if len(rows) > 0 else np.zeros((0, 5))


class FreezeFrameAutoencoderDataset(Dataset):

    def __init__(self, df):
        if "freeze_frame" not in df.columns:
            raise ValueError("Missing freeze_frame column")

        self.df = (
            df[df["freeze_frame"].notna()]
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return freeze_frame_to_array(
            row["freeze_frame"],
            flip=(row["team_id"] != row["home_team_id"]),
        )



def collate_freeze_frame_autoencoder(batch):
    batch_size = len(batch)
    max_players = max(x.shape[0] for x in batch)

    x = torch.zeros(batch_size, max_players, 5, dtype=torch.float32)
    mask = torch.ones(batch_size, max_players, dtype=torch.bool)

    for i, ff in enumerate(batch):
        n = ff.shape[0]
        x[i, :n] = torch.from_numpy(ff)
        mask[i, :n] = False

    return x, mask

class FreezeFrameSetEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, embedding_dim=32,
                 num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.pool_seed = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.pool_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, mask):
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=mask)

        b = x.size(0)
        seed = self.pool_seed.expand(b, -1, -1)

        pooled, _ = self.pool_attn(
            seed, h, h,
            key_padding_mask=mask
        )

        return self.out(pooled.squeeze(1))

class FreezeFramePointAutoencoder(nn.Module):
    def __init__(
        self,
        embedding_dim=32,
        max_players=22,
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.max_players = max_players

        self.encoder = FreezeFrameSetEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_players * 5),
        )

    def forward(self, x, mask):
        z = self.encoder(x, mask)

        out = self.decoder(z)
        out = out.view(-1, self.max_players, 5)

        return out, z


def batched_chamfer_loss(pred, target, mask, xy_weight, flag_weight):

    pred_xy = pred[:, :, :2]
    target_xy = target[:, :, :2]

    pred_flags = pred[:, :, 2:]
    target_flags = target[:, :, 2:]

    with torch.amp.autocast("cuda", enabled=False):
        dist = torch.cdist(pred_xy.float(), target_xy.float()) ** 2

    dist = dist.masked_fill(mask.unsqueeze(1), 1e6)

    p2t, idx = dist.min(dim=2)
    p2t = p2t.mean(dim=1)

    t2p, _ = dist.min(dim=1)

    valid = (~mask).float()
    t2p = (t2p * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    xy_loss = (p2t + t2p).mean()

    matched_flags = torch.gather(
        target_flags,
        1,
        idx.unsqueeze(-1).expand(-1, -1, target_flags.size(-1))
    )

    flag_loss = F.binary_cross_entropy_with_logits(
        pred_flags,
        matched_flags
    )

    return xy_weight * xy_loss + flag_weight * flag_loss