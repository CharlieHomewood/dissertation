from tqdm import tqdm
import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl
import socceraction.atomic.vaep.labels as atomic_vaep_labels
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*A value is trying to be set on a copy of a DataFrame.*"
)

def events_parquet_to_actions_df(events_parquet_file: str) -> pd.DataFrame:
    
    print(f"Reading in {events_parquet_file}")
    df = pd.read_parquet(events_parquet_file)
    print(f"Finished reading in {events_parquet_file}")

    match_ids = df["match_id"].unique().tolist()

    freeze_frame_lookup = (
        df[["event_uuid", "freeze_frame"]]
        .dropna(subset=["event_uuid"])
        .drop_duplicates(subset=["event_uuid"])
    )

    actions_dfs_list = []

    for match_id in tqdm(match_ids, desc="Processing Matches"):

        match_events = df[df["match_id"] == match_id]

        home_team_id = match_events["home_team_id"].iloc[0]

        actions = spadl.statsbomb.convert_to_actions(
            match_events,
            home_team_id,
            xy_fidelity_version=2,
            shot_fidelity_version=2,
        )

        atomic_actions = atomicspadl.convert_to_atomic(actions)

        atomic_actions_with_freeze_frame = atomic_actions.merge(
            freeze_frame_lookup,
            left_on="original_event_id",
            right_on="event_uuid",
            how="left",
            validate="many_to_one",
        )

        atomic_actions_with_freeze_frame["home_team_id"] = home_team_id

        labels = pd.concat([
            atomic_vaep_labels.scores(atomic_actions_with_freeze_frame),
            atomic_vaep_labels.concedes(atomic_actions_with_freeze_frame),
        ], axis=1)

        atomic_actions_with_freeze_frame = pd.concat(
            [atomic_actions_with_freeze_frame.reset_index(drop=True), labels.reset_index(drop=True)],
            axis=1,
        )

        actions_dfs_list.append(atomic_actions_with_freeze_frame)

    actions_df = pd.concat(actions_dfs_list, ignore_index=True)

    return actions_df