import numpy as np

from joblib import Parallel, delayed
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch, check_random_state, check_array


def _calculate_permutation_scores(
    estimator, X, y, col_idx, random_state, n_repeats, scorer
):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)

    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map
    # (memmap). X.copy() on the other hand is always guaranteed to return a
    # writable data-structure whose columns can be shuffled inplace.
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted.iloc[:, col_idx] = col
        else:
            shuffling_idx = shuffling_idx.reshape(X.shape[0], 1)
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx].reshape(
                X.shape[0], len(col_idx)
            )
        feature_score = scorer(estimator, X_permuted, y)
        scores[n_round] = feature_score

    return scores


def permutation_importance(
    estimator,
    x,
    y,
    *,
    feature_groups=None,
    scoring=None,
    n_repeats=5,
    n_jobs=None,
    random_state=None,
):
    """
    Computes permutation feature importance for a given estimator.

    :param estimator: An estimator which has already been fitted and is compatible
        with scorer.
    :param x ndarray or DataFrame, shape (n_samples, n_features): Data on which the
        permutation performance is to be computed.
    :param y array-like, shape (n_samples, ) or (n_samples, n_classes): Targets in case
        of a supervised estimator.
    :param feature_groups dict of lists or None: If features are to be bundled
        together, this provides a mapping between group names and features. The keys
        are the group names, while the values are lists of ints (feature indices)
        or strings (column names in case x is a DataFrame). Features which do not
        fall into any group are assumed to form a one-feature group.
    :param scoring string, callable or None: Scorer to use.
    :param n_repeats int, default=5: Number of times to permute each feature.
    :param n_jobs int or None, default=None: The number of jobs to use
        for the computation. If None, a single job is used. If -1, all available
        processors are used.
    :param random_state int, RandomState instance, default=None: Pseudo-random number
        generator to control the permutations or sampling of each feature.
    """
    if not hasattr(x, "iloc"):
        x = check_array(x, force_all_finite="allow-nan", dtype=None)

    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    scorer = check_scoring(estimator, scoring=scoring)
    baseline_score = scorer(estimator, x, y)

    feature_groups = _ensure_groups_complete(x, feature_groups)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_permutation_scores)(
            estimator, x, y, group_idx, random_seed, n_repeats, scorer
        )
        for group_idx in feature_groups.values()
    )

    importances = baseline_score - np.array(scores)
    return Bunch(
        importances_mean=np.mean(importances, axis=1),
        importances_std=np.mean(importances, axis=1),
        importances=importances,
        groups=feature_groups,
    )


def _ensure_groups_complete(x, feature_groups):
    """
    Adds missing groups (if needed) to cover all features
    present in x.
    """
    if feature_groups is None:
        feature_groups = {}

    complete_feature_groups = feature_groups.copy()

    if feature_groups is None or len(feature_groups) == 0:
        if hasattr(x, "columns"):
            return {col: [i] for i, col in enumerate(x.columns)}
        else:
            return {str(i): [i] for i in range(x.shape[1])}

    covered = [f for group in feature_groups.values() for f in group]

    if isinstance(covered[0], str):
        all_features = list(x.columns)
    else:
        all_features = [i for i in range(x.shape[1])]

    missing_features = [f for f in all_features if f not in covered]
    for feature in missing_features:
        feature_idx = all_features.index(feature)
        new_group_name = str(feature_idx)
        while new_group_name in complete_feature_groups:
            new_group_name += "_"
        complete_feature_groups[new_group_name] = [feature_idx]

    # Make sure that features are defined by their indices
    complete_feature_groups = {
        group_name: [all_features.index(f) for f in grouped_features]
        for group_name, grouped_features in complete_feature_groups.items()
    }

    return complete_feature_groups
