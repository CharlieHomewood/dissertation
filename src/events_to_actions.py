from utils.events_to_actions import events_parquet_to_actions_df
from tqdm import tqdm
import os

os.makedirs("data/actions", exist_ok=True)

for events_parquet_file in tqdm(os.listdir("data/events"), desc="Processing events files"):

    df = events_parquet_to_actions_df(f"data/events/{events_parquet_file}")

    df.to_parquet(
        f"data/actions/{events_parquet_file.replace("events", "actions")}",
        compression="zstd",
        index=False
    )