import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.atomic.vaep.features as fs
import tqdm


def previous_action_features(df, cols, nb_prev_actions):
    feature_frames = []

    for i in range(nb_prev_actions):
        shifted = (
            df[cols]
            .shift(i)
            .add_suffix(f"_a{i}")
        )
        feature_frames.append(shifted)

    return pd.concat(feature_frames, axis=1)


def actions_parquet_to_features_and_labels(
    actions_parquet_file: str,
    xfns: list,
    yfns: list,
    nb_prev_actions: int,
    spatial_feature_cols: list[str] | None = None,
):
    actions = pd.read_parquet(actions_parquet_file)

    actions = atomicspadl.add_names(actions.copy())

    games = (
        actions.groupby("game_id", as_index=False)
        .agg(home_team_id=("home_team_id", "first"))
    )
    
    feature_parts = []
    label_parts = []

    VAEP_COLUMNS = [
        "game_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        "type_id",
        "type_name",
        "bodypart_id",
        "bodypart_name",
        "x",
        "y",
        "dx",
        "dy",
    ]

    for game in tqdm.tqdm(
        games.itertuples(index=False),
        total=len(games),
        desc="Features/labels",
    ):

        game_actions = actions[
            actions["game_id"] == game.game_id
        ].copy()

        vaep_actions = game_actions[VAEP_COLUMNS].copy()

        gamestates = fs.gamestates(
            vaep_actions,
            nb_prev_actions
        )

        gamestates = fs.play_left_to_right(
            gamestates, 
            game.home_team_id
        )

        # -------------------------
        # Original VAEP features
        # -------------------------
        feature_game = pd.concat(
            [fn(gamestates) for fn in xfns],
            axis=1,
        )

        # -------------------------
        # Manual spatial features +
        # embeddings over previous actions
        # -------------------------
        if spatial_feature_cols:

            spatial_features = previous_action_features(
                game_actions,
                spatial_feature_cols,
                nb_prev_actions,
            )

            feature_game = pd.concat(
                [
                    feature_game.reset_index(drop=True),
                    spatial_features.reset_index(drop=True),
                ],
                axis=1,
            )

        feature_game.insert(
            0,
            "game_id",
            game.game_id,
        )

        feature_game.insert(
            1,
            "action_id",
            game_actions["action_id"].to_numpy(),
        )

        # -------------------------
        # Labels
        # -------------------------
        label_game = pd.concat(
            [fn(vaep_actions) for fn in yfns],
            axis=1,
        )

        label_game.insert(
            0,
            "game_id",
            game.game_id,
        )

        label_game.insert(
            1,
            "action_id",
            game_actions["action_id"].to_numpy(),
        )

        feature_parts.append(feature_game)
        label_parts.append(label_game)

    feature_df = pd.concat(
        feature_parts,
        ignore_index=True,
    )

    label_df = pd.concat(
        label_parts,
        ignore_index=True,
    )

    assert feature_df[
        ["game_id", "action_id"]
    ].equals(
        label_df[
            ["game_id", "action_id"]
        ]
    )

    return feature_df, label_df