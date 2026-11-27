"""
run in terminal with python -m analysis.try_kde_bandwidth
"""

from pathlib import Path
import numpy as np
import pandas as pd

from analysis.config import EMBEDDINGS_DIR, PROJECT_ROOT
from analysis.running_dist import compute_running_dist, temporal_cv

bandwidths = [0.05, 0.1, 0.2, 0.3]

embeddings_path = EMBEDDINGS_DIR / "embeddings_resnet18.npy"
metadata_path = EMBEDDINGS_DIR / "embeddings_resnet18_meta.csv"

embeddings = np.load(embeddings_path)
metadata = pd.read_csv(metadata_path)    

for bw in bandwidths:
    scores = compute_running_dist(
        embeddings,
        metadata,
        window_days=30,
        k_neighbours=10,
        min_history=10,
        kde_bandwidth=bw,
    )
    print(
        f"Bandwidth: {bw}",
        f"Mean: {scores['kde_full'].mean():.3f}",
        f"Std: {scores['kde_full'].std():.3f}",
        f"Min: {scores['kde_full'].min():.3f}",
        f"Max: {scores['kde_full'].max():.3f}",
    )
