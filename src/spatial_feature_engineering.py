from tqdm import tqdm
from utils.spatial_features import actors_euclidean_distance_to_each_player, number_of_players_within_radius_of_actor, player_convex_hull_area
import numpy as np
import os
import pandas as pd

for actions_file in os.listdir("data/actions"):
    df = pd.read_parquet(f"data/actions/{actions_file}")

    opponent_minimum_distance_values = []
    teammate_minimum_distance_values = []
    opponent_mean_distance_values = []
    teammate_mean_distance_values = []
    opponent_stdev_distance_values = []
    teammate_stdev_distance_values = []
    opponent_radius_values = []
    teammate_radius_values = []
    opponent_hull_values = []
    teammate_hull_values = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Processing freeze frames",
    ):

        freeze_frame = row["freeze_frame"]

        flip = row["team_id"] != row["home_team_id"]
        
        distance_values = actors_euclidean_distance_to_each_player(
            freeze_frame,
            flip=flip,
        )

        if distance_values is None:
            opponent_minimum_distance_values.append(np.nan)
            teammate_minimum_distance_values.append(np.nan)
            opponent_mean_distance_values.append(np.nan)
            teammate_mean_distance_values.append(np.nan)
            opponent_stdev_distance_values.append(np.nan)
            teammate_stdev_distance_values.append(np.nan)
        else:
            opponent_distances = distance_values["opponent_distances_to_actor"]
            teammate_distances = distance_values["teammate_distances_to_actor"]

            if opponent_distances:
                opponent_minimum_distance_values.append(min(opponent_distances))
                opponent_mean_distance_values.append(np.mean(opponent_distances))
                opponent_stdev_distance_values.append(np.std(opponent_distances))
            else:
                opponent_minimum_distance_values.append(np.nan)
                opponent_mean_distance_values.append(np.nan)
                opponent_stdev_distance_values.append(np.nan)

            if teammate_distances:
                teammate_minimum_distance_values.append(min(teammate_distances))
                teammate_mean_distance_values.append(np.mean(teammate_distances))
                teammate_stdev_distance_values.append(np.std(teammate_distances))
            else:
                teammate_minimum_distance_values.append(np.nan)
                teammate_mean_distance_values.append(np.nan)
                teammate_stdev_distance_values.append(np.nan)

        radius_counts = number_of_players_within_radius_of_actor(
            freeze_frame,
            radius=5,
            flip=flip,
        )

        hull_areas = player_convex_hull_area(
            freeze_frame,
            flip=flip,
        )

        if radius_counts is None:
            opponent_radius_values.append(np.nan)
            teammate_radius_values.append(np.nan)
        else:
            opponent_radius_values.append(
                radius_counts["number_of_opponents_within_radius_of_actor"]
            )
            teammate_radius_values.append(
                radius_counts["number_of_teammates_within_radius_of_actor"]
            )

        if hull_areas is None:
            opponent_hull_values.append(np.nan)
            teammate_hull_values.append(np.nan)
        else:
            opponent_hull_values.append(
                hull_areas["opponent_convex_hull_area"]
            )
            teammate_hull_values.append(
                hull_areas["teammate_convex_hull_area"]
            )

    df["actor_minimum_distance_to_opponent"] = opponent_minimum_distance_values
    df["actor_minimum_distance_to_teammate"] = teammate_minimum_distance_values
    df["actor_mean_distance_to_opponent"] = opponent_mean_distance_values
    df["actor_mean_distance_to_teammate"] = teammate_mean_distance_values
    df["actor_stdev_distance_to_opponent"] = opponent_stdev_distance_values
    df["actor_stdev_distance_to_teammate"] = teammate_stdev_distance_values
    df["number_of_opponents_within_radius_of_actor"] = opponent_radius_values
    df["number_of_teammates_within_radius_of_actor"] = teammate_radius_values
    df["opponent_convex_hull_area"] = opponent_hull_values
    df["teammate_convex_hull_area"] = teammate_hull_values

    df.to_parquet(f"data/actions/{actions_file}", compression="zstd", index=False)