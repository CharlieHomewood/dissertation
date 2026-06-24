import numpy as np
from sklearn.metrics import check_scoring
from tqdm import tqdm

def grouped_permutation_importance(
    estimator,
    X,
    y,
    feature_groups,
    group_mode,
    scoring=None,
    n_repeats=10,
    random_state=None,
):
    rng = np.random.RandomState(random_state)
    scorer = check_scoring(estimator, scoring=scoring)

    baseline_score = scorer(estimator, X, y)

    results = {}

    total_steps = 0
    for group_name, cols in feature_groups.items():
        if group_mode[group_name] == "independent":
            total_steps += len(cols) * n_repeats
        elif group_mode[group_name] == "block":
            total_steps += n_repeats
        else:
            raise ValueError(
                f"group_mode['{group_name}'] must be "
                f"'independent' or 'block'."
            )

    with tqdm(total=total_steps, desc="Permutation importance") as pbar:

        for group_name, cols in feature_groups.items():

            mode = group_mode[group_name]

            if mode == "independent":

                for col in cols:

                    drops = []

                    for _ in range(n_repeats):

                        X_perm = X.copy()

                        perm = rng.permutation(len(X_perm))
                        X_perm[col] = X_perm[col].iloc[perm].values

                        score = scorer(estimator, X_perm, y)

                        drops.append(baseline_score - score)

                        pbar.update(1)

                    results[col] = {
                        "mean": np.mean(drops),
                        "std": np.std(drops, ddof=1),
                        "raw": np.asarray(drops),
                    }

            elif mode == "block":

                drops = []

                for _ in range(n_repeats):

                    X_perm = X.copy()

                    perm = rng.permutation(len(X_perm))
                    X_perm.loc[:, cols] = X_perm.loc[:, cols].iloc[perm].values

                    score = scorer(estimator, X_perm, y)

                    drops.append(baseline_score - score)

                    pbar.update(1)

                results[group_name] = {
                    "mean": np.mean(drops),
                    "std": np.std(drops, ddof=1),
                    "raw": np.asarray(drops),
                }

    return results