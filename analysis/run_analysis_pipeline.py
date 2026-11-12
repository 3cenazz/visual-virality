from pathlib import Path
import numpy as np
import pandas as pd

from analysis.config import EMBEDDINGS_DIR, DATASET_DIR
from analysis.running_dist import compute_running_dist

def main():
    embeddings_path = EMBEDDINGS_DIR / "embeddings_resnet18.npy"
    metadata_path = EMBEDDINGS_DIR / "embeddings_resnet18_meta.csv"

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)

    scored = compute_running_dist(
        embeddings,
        metadata,
        window_days=30,
        k_neighbors=10,
        min_history=10,
    )
    out_path = DATASET_DIR / "running_dist.csv"
    scored.to_csv(out_path, index=False)
    print(f"Saved running distribution: {out_path}")

if __name__ == "__main__":
    main()
