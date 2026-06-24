import os
import copy
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, ParameterSampler
from scipy.stats import loguniform
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.spatial_embeddings import (
    FreezeFrameAutoencoderDataset,
    FreezeFramePointAutoencoder,
    collate_freeze_frame_autoencoder,
    batched_chamfer_loss,
)

# =============================================================================
# CONFIG
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

n_trials = 50
max_epochs = 50
patience = 5
seed = 42

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

best_model_path = f"{model_dir}/best_set_autoencoder.pt"
loss_plot_path = f"{model_dir}/best_loss_curve.png"
search_results_path = f"{model_dir}/hyperparameter_search.csv"


# =============================================================================
# DATA LOADING
# =============================================================================

dfs = []
for f in os.listdir("data/actions"):
    dfs.append(pd.read_parquet(f"data/actions/{f}"))

df = pd.concat(dfs, ignore_index=True)

match_col = "game_id" if "game_id" in df.columns else "match_id"
match_ids = df[match_col].dropna().unique()

train_val_ids, test_ids = train_test_split(
    match_ids, test_size=0.2, random_state=seed
)

train_ids, val_ids = train_test_split(
    train_val_ids, test_size=0.125, random_state=seed
)

train_df = df[df[match_col].isin(train_ids)]
val_df = df[df[match_col].isin(val_ids)]


train_ds = FreezeFrameAutoencoderDataset(train_df)
val_ds = FreezeFrameAutoencoderDataset(val_df)


# =============================================================================
# TRAIN FUNCTION (single run)
# =============================================================================

def train_model(params):

    train_loader = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_freeze_frame_autoencoder,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_freeze_frame_autoencoder,
    )

    model = FreezeFramePointAutoencoder(
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["wd"],
    )

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    train_curve = []
    val_curve = []

    for epoch in range(max_epochs):

        # -------------------
        # TRAIN
        # -------------------
        model.train()
        train_loss = 0

        for x, mask in train_loader:
            x, mask = x.to(device), mask.to(device)

            optimizer.zero_grad()
            pred, _ = model(x, mask)

            loss = batched_chamfer_loss(pred, x, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # -------------------
        # VALIDATION
        # -------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, mask in val_loader:
                x, mask = x.to(device), mask.to(device)

                pred, _ = model(x, mask)
                loss = batched_chamfer_loss(pred, x, mask)

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        train_curve.append(train_loss)
        val_curve.append(val_loss)

        # -------------------
        # EARLY STOPPING
        # -------------------
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)

    return model, best_val, train_curve, val_curve


# =============================================================================
# RANDOM SEARCH SPACE
# =============================================================================

search_space = {
    "embedding_dim": [16, 32, 64],
    "hidden_dim": [64, 128],
    "num_heads": [2, 4, 8],
    "num_layers": [1, 2, 3],
    "batch_size": [512, 1024],
    "lr": loguniform(1e-4, 1e-2),
    "wd": loguniform(1e-6, 1e-3),
}

samples = list(ParameterSampler(
    search_space,
    n_iter=n_trials,
    random_state=seed
))


# =============================================================================
# RANDOM SEARCH LOOP
# =============================================================================

results = []

best_global = float("inf")
best_model = None
best_params = None
best_train_curve = None
best_val_curve = None

for i, params in enumerate(tqdm(samples, desc="Random search")):

    print(f"\nTrial {i+1}/{n_trials}")
    print(params)

    model, val_loss, train_curve, val_curve = train_model(params)

    results.append({
        **params,
        "best_val_loss": val_loss
    })

    if val_loss < best_global:
        best_global = val_loss
        best_model = copy.deepcopy(model.state_dict())
        best_params = params
        best_train_curve = train_curve
        best_val_curve = val_curve

        print(f"🔥 New best: {best_global:.6f}")


# =============================================================================
# SAVE SEARCH RESULTS
# =============================================================================

pd.DataFrame(results).sort_values("best_val_loss").to_csv(
    search_results_path,
    index=False
)


# =============================================================================
# SAVE BEST MODEL
# =============================================================================

torch.save(
    {
        "model_state_dict": best_model,
        **best_params,
        "best_val_loss": best_global,
    },
    best_model_path
)


# =============================================================================
# PLOT BEST CURVE
# =============================================================================

plt.figure()

plt.plot(best_train_curve, label="Train")
plt.plot(best_val_curve, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Chamfer loss")
plt.title("Best Model Training Curve (Random Search)")
plt.legend()

plt.tight_layout()
plt.savefig(loss_plot_path, dpi=300)
plt.close()


print("Done.")
print("Best val loss:", best_global)
print("Best params:", best_params)