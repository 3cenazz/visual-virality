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

    # Temporal CV on all images
    cv_all = temporal_cv(
        embeddings,
        metadata,
    )
    cv_all.to_csv(
         PROJECT_ROOT / "temporal_cv_all.csv",
         index=False
         )
    print("CV All" + "="*10)
    print(cv_all.head(10))

    # Image category
    cv_cat = temporal_cv(
        embeddings,
        metadata,
        group_by="category"
    )
    cv_cat.to_csv(
         PROJECT_ROOT / "temporal_cv_category.csv",
         index=False
         )
    print("CV Category" + "="*10)
    print(cv_cat.head(10))

    # Publisher
    cv_pub = temporal_cv(
        embeddings,
        metadata,
        group_by="publisher"
    )
    cv_pub.to_csv(
         PROJECT_ROOT / "temporal_cv_publisher.csv",
         index=False
         )
    print("CV Publisher" + "="*10)
    print(cv_pub.head(10))

if __name__ == "__main__":
    main()