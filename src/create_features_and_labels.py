from utils.create_features_and_labels import actions_parquet_to_features_and_labels

import os
import socceraction.atomic.vaep.features as fs
import socceraction.atomic.vaep.labels as lab


dirs = ["data/features", "data/labels"]

for directory in dirs:
    os.makedirs(directory, exist_ok=True)


xfns = [
    fs.actiontype_onehot,
    fs.bodypart_onehot,
    fs.goalscore,
    fs.location,
    fs.polar,
    fs.direction,
    fs.team,
    fs.time,
    fs.time_delta,
]


yfns = [
    lab.scores,
    lab.concedes,
    lab.goal_from_shot,
]

spatial_feature_cols = [
    "actor_minimum_distance_to_opponent",
    "actor_minimum_distance_to_teammate",
    "actor_mean_distance_to_opponent",
    "actor_mean_distance_to_teammate",
    "actor_stdev_distance_to_opponent",
    "actor_stdev_distance_to_teammate",
    "number_of_opponents_within_radius_of_actor",
    "number_of_teammates_within_radius_of_actor",
    "opponent_convex_hull_area",
    "teammate_convex_hull_area"
]

embedding_feature_cols = [
    f"spatial_embedding_{i:02d}"
    for i in range(32)
]


for actions_parquet_file in os.listdir("data/actions"):
    if not actions_parquet_file.endswith(".parquet"):
        continue

    actions_path = f"data/actions/{actions_parquet_file}"

    feature_df, labels_df = actions_parquet_to_features_and_labels(
        actions_parquet_file=actions_path,
        xfns=xfns, 
        yfns=yfns,
        nb_prev_actions=3,
        spatial_feature_cols=spatial_feature_cols + embedding_feature_cols,
    )

    feature_output_file = actions_parquet_file.replace(
        "actions",
        "features"
    )

    labels_output_file = actions_parquet_file.replace(
        "actions",
        "labels"
    )

    feature_df.to_parquet(
        f"data/features/{feature_output_file}",
        compression="zstd",
        index=False
    )

    labels_df.to_parquet(
        f"data/labels/{labels_output_file}",
        compression="zstd",
        index=False
    )

    print(
        actions_parquet_file,
        feature_df.shape,
        labels_df.shape
    )