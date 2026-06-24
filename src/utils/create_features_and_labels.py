import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.atomic.vaep.features as fs
import tqdm
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns",
    category=FutureWarning,
)

def actions_parquet_to_features_and_labels(
    actions_parquet_file: str,
    xfns: list,
    yfns: list,
    nb_prev_actions: int,
    spatial_feature_cols: list[str] | None = None
):
    actions = pd.read_parquet(actions_parquet_file)

    actions = atomicspadl.add_names(actions)

    games = (
        actions.groupby("game_id", as_index=False)
        .agg(home_team_id=("team_id", "first"))
    )

    feature_parts = []
    label_parts = []

    for game in tqdm.tqdm(
        games.itertuples(index=False),
        total=len(games),
        desc="Features/labels"
    ):
        game_actions = actions[actions["game_id"] == game.game_id].copy()

        # Keep custom spatial features separately
        if spatial_feature_cols is not None:
            spatial_feature_game = game_actions[
                spatial_feature_cols
            ].reset_index(drop=True)
        else:
            spatial_feature_game = None

        # Remove columns that socceraction gamestates cannot handle
        cols_to_exclude_from_gamestates = ["freeze_frame"]

        if spatial_feature_cols is not None:
            cols_to_exclude_from_gamestates += spatial_feature_cols

        game_actions_for_vaep = game_actions.drop(
            columns=[
                col for col in cols_to_exclude_from_gamestates
                if col in game_actions.columns
            ]
        )

        gamestates = fs.gamestates(
            game_actions_for_vaep,
            nb_prev_actions
        )

        gamestates = fs.play_left_to_right(
            gamestates,
            game.home_team_id
        )

        feature_game = pd.concat(
            [fn(gamestates) for fn in xfns],
            axis=1
        )

        # Add scalar spatial features after normal VAEP features
        if spatial_feature_game is not None:
            feature_game = pd.concat(
                [
                    feature_game.reset_index(drop=True),
                    spatial_feature_game.reset_index(drop=True)
                ],
                axis=1
            )

        feature_game.insert(
            0,
            "game_id",
            game.game_id
        )

        feature_game.insert(
            1,
            "action_id",
            game_actions["action_id"].to_numpy()
        )

        label_game = pd.concat(
            [fn(game_actions_for_vaep) for fn in yfns],
            axis=1
        )

        label_game.insert(
            0,
            "game_id",
            game.game_id
        )

        label_game.insert(
            1,
            "action_id",
            game_actions["action_id"].to_numpy()
        )

        feature_parts.append(feature_game)
        label_parts.append(label_game)

    feature_df = pd.concat(feature_parts, ignore_index=True)
    label_df = pd.concat(label_parts, ignore_index=True)

    assert feature_df[["game_id", "action_id"]].equals(
        label_df[["game_id", "action_id"]]
    )

    return feature_df, label_df