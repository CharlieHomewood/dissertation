from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.spatial_embeddings import (
    FreezeFrameAutoencoderDataset,
    FreezeFramePointAutoencoder,
    collate_freeze_frame_autoencoder,
    batched_chamfer_loss,
)
import matplotlib.pyplot as plt
import optuna
import os
import pandas as pd
import pickle
import torch

model_path = "models/freeze_frame_point_autoencoder.pt"
seed = 123

num_epochs = 10
best_num_epochs = 20
max_players = 22
num_trials = 50

os.makedirs("models", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

action_paths = ["data/actions/" + file for file in os.listdir("data/actions")]

dfs = []
for path in action_paths:
    df_part = pd.read_parquet(path)
    df_part["source_file"] = path
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

match_col = "game_id" if "game_id" in df.columns else "match_id"
match_ids = df[match_col].dropna().unique()

train_val_ids, test_ids = train_test_split(
    match_ids, test_size=0.2, random_state=seed, shuffle=True
)
train_ids, val_ids = train_test_split(
    train_val_ids, test_size=0.125, random_state=seed, shuffle=True
)

train_df = df[df[match_col].isin(train_ids)].copy()
val_df = df[df[match_col].isin(val_ids)].copy()
test_df = df[df[match_col].isin(test_ids)].copy()

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

train_dataset = FreezeFrameAutoencoderDataset(train_df)
val_dataset = FreezeFrameAutoencoderDataset(val_df)
test_dataset = FreezeFrameAutoencoderDataset(test_df)

with open("data/set_transformer/train_dataset.pkl", "wb") as handle:
    pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/set_transformer/val_dataset.pkl", "wb") as handle:
    pickle.dump(val_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/set_transformer/test_dataset.pkl", "wb") as handle:
    pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

def objective(trial):

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
    embedding_dim = trial.suggest_categorical("embedding_dim", [16, 32, 64])
    flag_weight = trial.suggest_float("flag_weight", 0.05, 0.5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3, 4])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    xy_weight = trial.suggest_float("xy_weight", 0.5, 2.0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_freeze_frame_autoencoder,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_freeze_frame_autoencoder,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = FreezeFramePointAutoencoder(
        embedding_dim=embedding_dim,
        max_players=max_players,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scaler = torch.amp.GradScaler(device="cuda")

    best_val = float("inf")

    for epoch in tqdm(range(num_epochs), desc=f"Trial {trial.number}", leave=False):

        model.train()
        train_loss = 0.0

        for x, mask in train_loader:

            x = x.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda"):
                recon, _ = model(x, mask)

                loss = batched_chamfer_loss(
                    recon,
                    x,
                    mask,
                    xy_weight=xy_weight,
                    flag_weight=flag_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0

        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            for x, mask in val_loader:

                x = x.to(device)
                mask = mask.to(device)

                recon, _ = model(x, mask)

                loss = batched_chamfer_loss(
                    recon,
                    x,
                    mask,
                    xy_weight=xy_weight,
                    flag_weight=flag_weight,
                )

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val = min(best_val, val_loss)

    return best_val

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=2,
        max_resource=num_epochs,
        reduction_factor=2
    ),
)

study.optimize(objective, n_trials=num_trials)

results_df = study.trials_dataframe()

results_df.to_parquet(
    "results/optuna_results.parquet",
    compression="zstd",
    index=False,
)

print("\nBEST TRIAL:")
print(study.best_trial.value)
print(study.best_trial.params)

best_params = study.best_trial.params


train_loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    collate_fn=collate_freeze_frame_autoencoder,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=best_params["batch_size"],
    shuffle=False,
    collate_fn=collate_freeze_frame_autoencoder,
)

model = FreezeFramePointAutoencoder(
    embedding_dim=best_params["embedding_dim"],
    max_players=max_players,
    hidden_dim=best_params["hidden_dim"],
    num_heads=best_params["num_heads"],
    num_layers=best_params["num_layers"],
    dropout=best_params["dropout"],
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=best_params["learning_rate"],
    weight_decay=best_params["weight_decay"],
)

train_losses, val_losses = [], []
best_state = None
best_val_loss = float("inf")

for epoch in tqdm(range(best_num_epochs), desc="Best model"):

    model.train()
    train_loss = 0.0

    for x, mask in train_loader:
        x = x.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            recon, _ = model(x, mask)

            loss = batched_chamfer_loss(
                recon,
                x,
                mask,
                xy_weight=best_params["xy_weight"],
                flag_weight=best_params["flag_weight"],
            )

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0

    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        for x, mask in val_loader:

            x = x.to(device)
            mask = mask.to(device)

            recon, _ = model(x, mask)

            loss = batched_chamfer_loss(
                recon,
                x,
                mask,
                xy_weight=best_params["xy_weight"],
                flag_weight=best_params["flag_weight"],
            )

            val_loss += loss.item() * x.size(0)

    val_loss /= len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

torch.save(
    {
        "model_state_dict": best_state,
        "best_params": best_params,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    },
    model_path,
)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.xlabel("Epoch")
plt.ylabel("Chamfer loss")
plt.legend()
plt.tight_layout()
plt.savefig("reports/dissertation/figures/freeze_frame_point_autoencoder_loss_curve.png", dpi=300)
plt.close()

model.load_state_dict(best_state)
model.eval()

test_loader = DataLoader(
    test_dataset,
    batch_size=best_params["batch_size"],
    shuffle=False,
    collate_fn=collate_freeze_frame_autoencoder,
)

test_loss = 0.0

with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
    for x, mask in test_loader:

        x = x.to(device)
        mask = mask.to(device)

        recon, _ = model(x, mask)

        loss = batched_chamfer_loss(
            recon,
            x,
            mask,
            xy_weight=best_params["xy_weight"],
            flag_weight=best_params["flag_weight"],
        )

        test_loss += loss.item() * x.size(0)

test_loss /= len(test_loader.dataset)

with open("results/set_transformer_test_loss.pkl", "wb") as handle:
    pickle.dump(test_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Test loss: {test_loss:.4f}")