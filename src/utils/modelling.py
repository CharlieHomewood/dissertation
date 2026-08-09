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

    columns = list(X.columns)
    col_to_idx = {col: i for i, col in enumerate(columns)}

    values = X.to_numpy(dtype=np.float64, copy=True)
    n_rows = values.shape[0]

    baseline_score = scorer(estimator, values, y)

    results = {}

    total_steps = 0
    for group_name, cols in feature_groups.items():
        mode = group_mode[group_name]
        if mode == "independent":
            total_steps += len(cols) * n_repeats
        elif mode == "block":
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
                    idx = col_to_idx[col]
                    original_col = values[:, idx].copy()
                    drops = np.empty(n_repeats)

                    for i in range(n_repeats):
                        perm = rng.permutation(n_rows)
                        values[:, idx] = original_col[perm]

                        score = scorer(estimator, values, y)
                        drops[i] = baseline_score - score

                        pbar.update(1)

                    values[:, idx] = original_col

                    results[col] = {
                        "mean": drops.mean(),
                        "std": drops.std(ddof=1),
                        "raw": drops,
                    }

            elif mode == "block":
                idxs = [col_to_idx[c] for c in cols]
                original_block = values[:, idxs].copy()
                drops = np.empty(n_repeats)

                for i in range(n_repeats):
                    perm = rng.permutation(n_rows)
                    values[:, idxs] = original_block[perm]

                    score = scorer(estimator, values, y)
                    drops[i] = baseline_score - score

                    pbar.update(1)

                values[:, idxs] = original_block

                results[group_name] = {
                    "mean": drops.mean(),
                    "std": drops.std(ddof=1),
                    "raw": drops,
                }

    return results