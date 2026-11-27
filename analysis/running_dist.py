import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

window_days = 30  # rolling window days
knn = 10          # K Nearest Neighbour images
min_history = 10  # minimum number of historical images
softmax_temp = 0.1
kde_bandwidth = 0.1

def compute_running_dist(
    embeddings,
    metadata,
    window_days=window_days,
    k_neighbors=knn,
    min_history=min_history,
    softmax_temperature=softmax_temp ,
    kde_bandwidth=kde_bandwidth,
):
    """
    Compute image surprisal metrics against images in the preceding time window.

    Reference sets:
      - full: all images in the preceding window
      - knn10 (or k_neighbors): k nearest historical images
      - knn1: single closest historical image

    Measures:
      - cosine distance mean/min
      - softmax entropy over cosine (nats): distribution of cosine similarity
      High value means the image has many historical matches, low value means only a small few dominate.
      - exponential-kernel density score: -log(mean(exp(-d / bandwidth))): similarity-density proxy
      Larger values mean lower density => greater novelty.

    Same-day rows are excluded since we don't have within-day distinctions.
    """
    if len(embeddings) != len(metadata):
        raise ValueError("embeddings and metadata must have equal row counts.")
    if softmax_temperature <= 0:
        raise ValueError("softmax_temperature must be > 0.")
    if kde_bandwidth <= 0:
        raise ValueError("kde_bandwidth must be > 0.")
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1.")

    df = metadata.copy().reset_index(drop=True)
    df["publication_date"] = pd.to_datetime(df["publication_date"])
    order = np.argsort(df["publication_date"].values)

    # L2-normalized embeddings (so we can use dot product for cosine similarity)
    emb = normalize(np.asarray(embeddings, dtype=float))
    n_rows = len(df)

    columns = [
        "cosine_full_mean",
        "cosine_knn10_mean",
        "cosine_knn1",
        "softmax_full",
        "softmax_knn10",
        "softmax_knn1",
        "kde_full",
        "kde_knn10",
        "kde_knn1",
        "history_count",
    ]
    results = {col: np.full(n_rows, np.nan, dtype=float) for col in columns}

    dates = df["publication_date"]

    for idx in order:
        image_date = dates.iloc[idx]
        start_date = image_date - pd.Timedelta(days=window_days)

        history_mask = (dates >= start_date) & (dates < image_date)
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
        k = min(k_neighbors, history_count)
        nearest_local = np.argpartition(distances, k - 1)[:k] # using this for performance, instead of argsort directly
        nearest_local = nearest_local[np.argsort(distances[nearest_local])]
        knn_distances = distances[nearest_local]

        results["cosine_knn10_mean"][idx] = float(np.mean(knn_distances))
        results["cosine_knn1"][idx] = float(knn_distances[0])

        # Softmax entropy: calculate cosine similarity, softmax it (becomes probabilities), sub into Shannon's Entropy
        # Compute for full set, k nearest, and nearest one.
        def softmax_entropy(sim_values):
            logits = sim_values / softmax_temperature
            logits = logits - np.max(logits)  # for numerical stability
            numer = np.exp(logits)
            denom = np.sum(numer)
            probs = numer/denom
            result = float(-np.sum(probs * np.log(probs + 1e-12)))
            return result

        results["softmax_full"][idx] = softmax_entropy(similarities)
        results["softmax_knn10"][idx] = softmax_entropy(similarities[nearest_local])
        results["softmax_knn1"][idx] = softmax_entropy(similarities[nearest_local[:1]])

        # Kernel Density Estimation
        # density proxy = mean(exp(-cosine_distance / bandwidth))
        # surprisal score = -log(density proxy)
        def kde(dist_values):
            kernel_values = np.exp(-dist_values / kde_bandwidth)
            density = float(np.mean(kernel_values))
            surprisal = float(-np.log(max(density, 1e-12)))
            return surprisal

        results["kde_full"][idx] = kde(distances)
        results["kde_knn10"][idx] = kde(knn_distances)
        results["kde_knn1"][idx] = kde(knn_distances[:1])

    out = df.copy()
    for col, values in results.items():
        if col == "history_count":
            out[col] = np.nan_to_num(values, nan=0).astype(int)
        else:
            out[col] = values

    # out["running_window_days"] = window_days
    # out["softmax_temperature"] = softmax_temperature
    # out["kde_bandwidth"] = kde_bandwidth
    return out