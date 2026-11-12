"""
Helper functions for pipeline
Ciculation procies, surprisal score, dataset stuff
"""

import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

def add_circulation_proxy(metadata, image_entity_col="image_entity_id"):
    """
    Estimate circulation at image/entity level from distinct dated source issues.
    Requires image_entity_id from a duplicate/repeated-image clustering step.
    Without it, this function treats each crop as a separate entity (not useful
    for repeat circulation); create entity IDs first.
    """
    df = metadata.copy()
    if image_entity_col not in df.columns:
        raise ValueError(f"Missing {image_entity_col}. Assign duplicate clusters first.")
    if "source_issue_id" not in df.columns:
        df["source_issue_id"] = df["source_filename"].astype(str)
    if "publication_date" not in df.columns:
        raise ValueError("publication_date is required")

    stats = df.groupby(image_entity_col).agg(
        appearance_count=("image_entity_id", "size"),
        distinct_issue_count=("source_issue_id", "nunique"),
        distinct_page_count=("source_page_id", "nunique"),
        first_seen=("publication_date", "min"),
        last_seen=("publication_date", "max"),
    ).reset_index()
    stats["circulation_proxy"] = np.log1p(stats["distinct_issue_count"])
    stats["repeat_appearance_proxy"] = np.log1p(stats["appearance_count"])
    stats["temporal_span_days"] = (
        pd.to_datetime(stats["last_seen"]) - pd.to_datetime(stats["first_seen"])
    ).dt.days
    return df.merge(stats, on=image_entity_col, how="left")

def calculate_reddit_inspired_score(
    metadata,
    image_entity_col="image_entity_id",
    issue_col="source_issue_id",
    date_col="publication_date",
):
    """
    Newspaper analogue inspired by Reddit's repost/cross-user circulation logic:
    score = log(1 + distinct issues containing entity) * temporal persistence factor.
    This is a proxy, not true readership/engagement virality.
    """
    if image_entity_col not in metadata:
        raise ValueError(f"Metadata needs {image_entity_col} from duplicate clustering.")
    df = metadata.copy()
    counts = df.groupby(image_entity_col).agg(
        distinct_issues=(issue_col, "nunique"),
        first_seen=(date_col, "min"),
        last_seen=(date_col, "max"),
        appearances=(image_entity_col, "size"),
    ).reset_index()
    span = (pd.to_datetime(counts["last_seen"]) -
            pd.to_datetime(counts["first_seen"])).dt.days
    # Bounded persistence multiplier: from 1 to 2 as span grows toward one year.
    persistence = 1 + np.minimum(span.clip(lower=0) / 365.0, 1.0)
    counts["circulation_score"] = np.log1p(counts["distinct_issues"]) * persistence
    counts["distinct_issues"] = counts["distinct_issues"].astype(int)
    return df.merge(counts, on=image_entity_col, how="left")

def image_surprisal_nn(query_vector, historical_vectors, k=10):
    """
    Computational novelty/surprisal proxy:
      S(x_t) = mean of k nearest cosine distances to historical context.
    Higher values indicate less similar historical context.
    This is not calibrated -log probability.
    """
    if len(historical_vectors) == 0:
        return np.nan
    hist = normalize(np.asarray(historical_vectors, dtype=np.float32))
    query = normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
    k = min(k, len(hist))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(hist)
    distances, _ = nn.kneighbors(query)
    return float(np.mean(distances[0]))

def image_surprisal_category(category, historical_categories, smoothing=1.0):
    """Smoothed categorical rarity: -log((count(category)+alpha)/(N+alpha*K))."""
    hist = list(historical_categories)
    if not hist:
        return np.nan
    categories = set(hist) | {category}
    count = sum(x == category for x in hist)
    return float(-np.log((count + smoothing) / (len(hist) + smoothing * len(categories))))

def exact_image_hash(image_path):
    """SHA256 hash of file bytes; detects byte-identical images only."""
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_vectors(embeddings):
    return normalize(np.asarray(embeddings, dtype=np.float32))
