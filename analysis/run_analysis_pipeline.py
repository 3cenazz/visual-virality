from pathlib import Path
import numpy as np
import pandas as pd

from analysis.config import EMBEDDINGS_DIR, PROJECT_ROOT
from analysis.running_dist import compute_running_dist, temporal_cv


def main():
    embeddings_path = EMBEDDINGS_DIR / "embeddings_resnet18.npy"
    metadata_path = EMBEDDINGS_DIR / "embeddings_resnet18_meta.csv"

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)        

    # Overall running distribution (no grouping)
    all_scores = compute_running_dist(
        embeddings,
        metadata,
        window_days=30,
        k_neighbours=10,
        min_history=10,
    )
    all_out_path = PROJECT_ROOT / "running_dist_all.csv"
    all_scores.to_csv(all_out_path, index=False)
    print(f"Saved running distribution (all): {all_out_path}")

    # Image category
    cat_scores = compute_running_dist(
        embeddings,
        metadata,
        window_days=30,
        k_neighbours=10,
        min_history=10,
        group_by="category"
    )
    cat_out_path = PROJECT_ROOT / "running_dist_category.csv"
    cat_scores.to_csv(cat_out_path, index=False) 
    print(f"Saved running distribution (category): {cat_out_path}")

    # Publisher
    pub_scores = compute_running_dist(
        embeddings,
        metadata,
        window_days=30,
        k_neighbours=10,
        min_history=10,
        group_by="publisher"
    )
    pub_out_path = PROJECT_ROOT / "running_dist_publisher.csv"
    pub_scores.to_csv(pub_out_path, index=False)   
    print(f"Saved running distribution (publisher): {pub_out_path}")

if __name__ == "__main__":
    main()
