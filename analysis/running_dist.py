"""
Compute image surprisal metrics against images in the preceding time window.

Reference sets:
- full: all images in the preceding window
- knn10 (or k_neighbors): k nearest historical images
- knn1: single closest historical image

In compute_running_dist
group_by: str or None
    None: uses all historial images
    "category": use only images with the same image category
    "publisher": use only images with the same publisher

Measures:
- cosine distance mean/min
- softmax entropy over cosine (nats): distribution of cosine similarity
    High value means the image has many historical matches, low value means only a small few dominate.
- exponential-kernel density score: -log(mean(exp(-d / bandwidth))): similarity-density proxy
    Larger values mean lower density => greater novelty.

Same-day rows are excluded since we don't have within-day distinctions.
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.stats import spearmanr

def softmax_entropy(sim, temp):
    """
    Softmax entropy: calculate cosine similarity, softmax it (becomes probabilities), sub into Shannon's Entropy
    Compute for full set, k nearest, and nearest one.
    """
    logits = sim / temp
    logits = logits - np.max(logits)  # for numerical stability
    numer = np.exp(logits)
    denom = np.sum(numer)
    probs = numer/denom
    result = float(-np.sum(probs * np.log(probs + 1e-12)))
    return result

def kde_surprisal(distance, bw):
    """
    Kernel Density Estimation:
    density proxy = mean(exp(-cosine_distance / bandwidth))
    surprisal score = -log(density proxy)
    """
    kernel_values = np.exp(-distance / bw)
    density = float(np.mean(kernel_values))
    surprisal = float(-np.log(max(density, 1e-12)))
    return surprisal


def compute_running_dist(
    embeddings,
    metadata,
    window_days=30,
    k_neighbours=10,
    min_history=10,
    softmax_temp=0.1,
    kde_bandwidth=0.05,
    group_by=None,
):
    if len(embeddings) != len(metadata):
        raise ValueError("embeddings and metadata must have equal row counts.")
    if softmax_temp <= 0:
        raise ValueError("softmax_temp must be > 0.")
    if kde_bandwidth <= 0:
        raise ValueError("kde_bandwidth must be > 0.")
    if k_neighbours < 1:
        raise ValueError("k_neighbours must be >= 1.")
    if window_days <= 0:
        raise ValueError("window_days must be > 0.")
    if min_history < 1:
        raise ValueError("min_history must be >= 1.")
    if group_by is not None and group_by not in metadata.columns:
        raise ValueError(f"Invalid grouping column: {group_by}")

    df = metadata.copy().reset_index(drop=True)
    df["publication_date"] = pd.to_datetime(df["publication_date"])

    # L2-normalized embeddings (so we can use dot product for cosine similarity)
    emb = normalize(np.asarray(embeddings, dtype=float))
    n_rows = len(df)

    columns = [
        "cosine_full_mean",
        f"cosine_knn{k_neighbours}_mean",
        "cosine_knn1",
        "softmax_full",
        f"softmax_knn{k_neighbours}",
        "kde_full",
        f"kde_knn{k_neighbours}",
        "kde_knn1",
        "history_count",
    ]
    results = {col: np.full(n_rows, np.nan, dtype=float) for col in columns}

    dates = df["publication_date"]

    if group_by is None:
        group_values = [None] * n_rows
    else:
        group_values = df[group_by].tolist()

    order = np.argsort(df["publication_date"].values)

    for idx in order:
        image_date = dates.iloc[idx]
        group_value = group_values[idx]

        if pd.isna(image_date):
            continue

        start_date = image_date - pd.Timedelta(days=window_days)

        history_mask = (dates >= start_date) & (dates < image_date)

        if group_by is not None:
            if pd.isna(group_value):
                continue
            history_mask &= df[group_by].eq(group_value)

        history_idx = np.flatnonzero(history_mask.to_numpy())

        history_count = len(history_idx)
        results["history_count"][idx] = history_count

        if history_count < min_history:
            continue

        query = emb[idx]
        hist = emb[history_idx]

        # Cosine similarity and distance (1 - similarity)
        similarities = np.clip(hist @ query, -1.0, 1.0) # dot product of L2-norm embeddings for cosine similarity
        distances = np.clip(1.0 - similarities, 0.0, 2.0)

        # Full history image set
        results["cosine_full_mean"][idx] = float(np.mean(distances))

        # KNN image set
        k = min(k_neighbours, history_count)
        nearest_idx = np.argpartition(distances, k - 1)[:k] # using this for performance, instead of argsort directly
        nearest_idx = nearest_idx[np.argsort(distances[nearest_idx])]
        knn_distances = distances[nearest_idx]
        knn_similarities = similarities[nearest_idx]

        results[f"cosine_knn{k_neighbours}_mean"][idx] = float(np.mean(knn_distances))
        results["cosine_knn1"][idx] = float(knn_distances[0])

        results["softmax_full"][idx] = softmax_entropy(similarities, softmax_temp)
        results[f"softmax_knn{k_neighbours}"][idx] = softmax_entropy(knn_similarities, softmax_temp)

        results["kde_full"][idx] = kde_surprisal(distances, kde_bandwidth)
        results[f"kde_knn{k_neighbours}"][idx] = kde_surprisal(knn_distances, kde_bandwidth)
        results["kde_knn1"][idx] = kde_surprisal(knn_distances[:1], kde_bandwidth)

    out = df.copy()
    for col, values in results.items():
        if col == "history_count":
            out[col] = np.nan_to_num(values, nan=0).astype(int)
        else:
            out[col] = values

    # out["running_window_days"] = window_days
    # out["running_k"] = k_neighbours
    # out["running_group"] = group_by or "all"
    return out


def temporal_cv(
    embeddings,
    metadata,
    k_values=(1, 5, 10, 20, 50),
    window_values=(1, 3, 7, 14, 30, 60, 90),
    min_history=10,
    kde_bandwidth=0.05,
    n_folds=5,
    group_by=None
):
    """
    Temporal cross validation to select k_neighbours and window_days.
    Since we have no ground truth labels, this is more to compare the 
    metrics computed against the full historical reference over the
    longest window (90 days). Running it over the full history is infeasible.
    
    Meant to observe stability: whether a shorter window and K preserve the
    surprisal ranking obtained from a larger historical context.

    Returns mean Spearman rank correlation.
    """
    
    df = metadata.copy().reset_index(drop=True)
    df["publication_date"] = pd.to_datetime(df["publication_date"])
    max_window = max(window_values)
    dates = df["publication_date"].dropna()

    # Reference: full historical set with the longest context window
    reference = compute_running_dist(
        embeddings,
        df,
        window_days=max_window,
        k_neighbours=max(k_values), # doesn't actually matter what you put here
        min_history=min_history,
        kde_bandwidth=kde_bandwidth,
        group_by=group_by
    )

    reference_scores = reference["kde_full"].to_numpy()
    unique_dates = np.sort(dates.unique())
    folds = np.array_split(unique_dates, n_folds)

    rows = []

    for window in window_values:
        for k in k_values:
            candidate = compute_running_dist(
                embeddings,
                df,
                window_days=window,
                k_neighbours=k,
                min_history=min_history,
                kde_bandwidth=kde_bandwidth,
                group_by=group_by
            )
            candidate_scores = candidate[f"kde_knn{k}"].to_numpy()
            fold_corrs = []
            for fold_dates in folds:
                if len(fold_dates) == 0:
                    continue
                fold_mask = df["publication_date"].isin(fold_dates).to_numpy()
                valid_mask = (
                    fold_mask
                    & np.isfinite(candidate_scores)
                    & np.isfinite(reference_scores)
                )
                if valid_mask.sum() < 3:
                    continue
                corr = spearmanr(
                    candidate_scores[valid_mask], 
                    reference_scores[valid_mask],
                ).statistic
                fold_corrs.append(float(corr))
            rows.append({
                "window_days": window,
                "k_neighbours": k,
                "mean_rank_corr": np.mean(fold_corrs) if fold_corrs else np.nan,
                "std_rank_corrs": np.std(fold_corrs) if fold_corrs else np.nan,
                "n_folds": len(fold_corrs)
            })
    results = pd.DataFrame(rows)
    return results.sort_values("mean_rank_corr", ascending=False).reset_index(drop=True)
