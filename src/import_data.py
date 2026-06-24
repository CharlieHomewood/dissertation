from concurrent.futures import ThreadPoolExecutor, as_completed
from socceraction.data.statsbomb import StatsBombLoader
from statsbombpy import sb
from tqdm import tqdm
from utils.data_importing import process_match
import os
import pandas as pd
import pickle
import requests
import warnings

warnings.filterwarnings("ignore")

dirs = ["data/metadata", "data/events"]
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

session = requests.Session()
SBL = StatsBombLoader()

if __name__ == "__main__":

    competitions_dict = {
        "mens_euro_2020": {"competition_id": 55, "season_id": 43},
        "mens_world_cup_2022": {"competition_id": 43, "season_id": 106},
        "mens_euro_2024": {"competition_id": 55, "season_id": 282},
        "womens_euro_2022": {"competition_id": 53, "season_id": 106},
        "womens_world_cup_2023": {"competition_id": 72, "season_id": 107},
        "womens_euro_2025": {"competition_id": 53, "season_id": 315}
    }

    number_of_events = 0
    number_of_matches = 0

    for comp_name, ids in competitions_dict.items():

        print(f"\nProcessing {comp_name}")

        matches_df = sb.matches(
            competition_id=ids["competition_id"],
            season_id=ids["season_id"]
        )

        games_df = SBL.games(
            competition_id=ids["competition_id"],
            season_id=ids["season_id"]
        )

        match_ids = matches_df["match_id"].unique()
        number_of_matches += len(match_ids)

        match_info_df = matches_df.merge(
            games_df,
            how="left",
            left_on="match_id",
            right_on="game_id"
        ).drop(columns=["game_id"])

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_match, match_id) for match_id in match_ids]

            for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading match data"):
                res = f.result()
                if res is not None:
                    results.append(res)

        statsbomb_events_list, socceraction_events_list, statsbomb_360_list = [], [], []

        for statsbomb_event, socceraction_event, statsbomb_360 in results:
            statsbomb_events_list.append(statsbomb_event)
            socceraction_events_list.append(socceraction_event)
            if statsbomb_360 is not None:
                statsbomb_360_list.append(statsbomb_360)

        print("Concatenating dataframes...")
        statsbomb_events_all = pd.concat(statsbomb_events_list, ignore_index=True)
        socceraction_events_all = pd.concat(socceraction_events_list, ignore_index=True)
        statsbomb_360_all = pd.concat(statsbomb_360_list, ignore_index=True) if statsbomb_360_list else pd.DataFrame()

        print("Merging into single dataframe...")
        df = statsbomb_events_all.merge(
            match_info_df,
            how="left",
            on="match_id"
        )

        df = df.merge(
            socceraction_events_all,
            how="left",
            left_on="id",
            right_on="event_id"
        )

        if not statsbomb_360_all.empty:
            df = df.merge(
                statsbomb_360_all,
                how="left",
                left_on="id",
                right_on="event_uuid"
            )

        out_path = f"data/events/events_{ids['competition_id']}_{ids['season_id']}.parquet"

        number_of_events += df.shape[0]

        df.to_parquet(out_path, index=False, compression="zstd")

        print(f"Saved to '~/{out_path}'")

    with open("data/metadata/number_of_matches.pkl", "wb") as f:
        pickle.dump(number_of_matches, f)
    
    with open("data/metadata/number_of_events.pkl", "wb") as f:
        pickle.dump(number_of_events, f)

    print(f"Finished! Downloaded and processed {number_of_events} events from {number_of_matches} matches across {len(competitions_dict.keys())} competitions.")