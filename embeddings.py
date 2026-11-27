from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
from tqdm.auto import tqdm
import time

from analysis.config import PROJECT_ROOT, CROPS_DIR, DATASET_DIR

ANALYSIS_DIR = DATASET_DIR / "analysis"
EMBEDDINGS_PATH = ANALYSIS_DIR / "embeddings.npy"
METADATA_PATH = ANALYSIS_DIR / "metadata.parquet"
MODEL_NAME = "resnet50_imagenet1k_v2"

# ImageNet pretrained ResNet50. Remove classifier to obtain 2048-D feature vectors.
def load_embedding_model(device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()
    return model, preprocess, device

def stable_image_id(relative_crop_path: str) -> str:
    """Stable key based on path; preserves distinct appearances of repeated imagery."""
    return hashlib.sha1(relative_crop_path.encode("utf-8")).hexdigest()

def load_crop_metadata():
    records = []

    metadata_dir = DATASET_DIR / "metadata" / "per_page"
    json_files = sorted(metadata_dir.glob("*.json"))
    print(f"Found {len(json_files):,} JSON files.", flush=True)
    if not json_files:
        raise ValueError(f"No JSON files found in {metadata_dir}")

    for json_path in tqdm(
        json_files,
        desc="Loading metadata files",
        unit="file",
        mininterval=0,
        miniters=1,
        disable=False,
    ):

        with json_path.open(encoding="utf-8") as f:
            page_records = json.load(f)

        for row in page_records:
            row = dict(row)
            rel_path = str(row["crop_filename"])
            row["image_id"] = stable_image_id(rel_path)
            row["image_path"] = str(CROPS_DIR / rel_path)
            row["source_issue_id"] = (
                f'{row.get("source_filename", "")}|{row.get("publication_date", "")}')
            row["source_page_id"] = (
                f'{row.get("source_filename", "")}|{row.get("page_number", "")}')
            records.append(row)
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No crop metadata found. Run the extraction pipeline first.")
    print(f"Total crop records: {len(df):,}", flush=True)
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df.dropna(subset=["publication_date"]).reset_index(drop=True)
    print(f"Valid metadata records (publication date): {len(df):,}", flush=True)
    return df

@torch.inference_mode()
def generate_embeddings(batch_size=32, device=None, overwrite=False):
    start_time = time.time()
    """Generate vectors aligned row-for-row with metadata.parquet."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    if (EMBEDDINGS_PATH.exists() and METADATA_PATH.exists() and not overwrite):
        print("Embeddings and metadata already exist; pass overwrite=True to rebuild.", flush=True)
        return (np.load(EMBEDDINGS_PATH), pd.read_parquet(METADATA_PATH))

    # Step 1: Load metadata
    print("\n[1/4] Loading crop metadata...", flush=True)
    df = load_crop_metadata()
    print(f"Total images to process: {len(df):,}", flush=True)

    # Step 2: Load model
    print("\n[2/4] Initialising embedding model...", flush=True)
    model, preprocess, device = load_embedding_model(device)

    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...", flush=True)
    vectors = []
    valid_rows = []
    num_batches = (len(df) + batch_size - 1) // batch_size
    skipped_images = 0
    batch_progress = tqdm(
        range(0, len(df), batch_size),
        total=num_batches,
        desc="Embedding batches",
        unit="batch",
    )

    for start in batch_progress:
        batch_df = df.iloc[start:start + batch_size]
       
        tensors = []
        indices = []

        for idx, row in tqdm(
            batch_df.iterrows(),
            total=len(batch_df),
            desc="Loading images",
            unit="image",
            leave=False,
        ):
            path = Path(row["image_path"])

            try:
                with Image.open(path) as im:
                    tensors.append(preprocess(im.convert("RGB")))
                indices.append(idx)
            except Exception as exc:
                skipped_images += 1
                tqdm.write(
                    f"Skipping unreadable crop {path}: {exc}"
                )

        if not tensors:
            batch_progress.set_postfix(
                processed=len(valid_rows),
                skipped=skipped_images,
            )
            continue

        # Stack image tensors and move them to the selected device
        batch = torch.stack(tensors).to(device)

        # Generate embeddings
        output = model(batch).cpu().numpy().astype(np.float32)

        vectors.extend(output)
        valid_rows.extend(indices)

        batch_progress.set_postfix(
            processed=f"{len(valid_rows):,}",
            skipped=skipped_images,
        )

    # Step 4: Align and save
    print("\n[4/4] Aligning metadata and saving outputs...", flush=True)
    metadata = df.loc[valid_rows].reset_index(drop=True)
    embeddings = np.asarray(vectors, dtype=np.float32)

    if len(metadata) != len(embeddings):
        raise RuntimeError("Embedding/metadata row alignment error.")

    print("Saving embeddings.npy...", flush=True)

    np.save(EMBEDDINGS_PATH, embeddings)

    print("Saving metadata.parquet...", flush=True)

    metadata.to_parquet(METADATA_PATH, index=False)

    elapsed = time.time() - start_time

    print("\nEmbedding generation completed!", flush=True)
    print(f"Total images processed: {len(embeddings):,}", flush=True)
    print(f"Skipped unreadable images: {skipped_images:,}", flush=True)
    print(f"Embedding shape: {embeddings.shape}", flush=True)
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}", flush=True)
    print(f"Saved metadata to: {METADATA_PATH}", flush=True)
    print(f"Total time: {elapsed / 60:.2f} minutes", flush=True)

    return embeddings, metadata


def load_embeddings():
    return np.load(EMBEDDINGS_PATH), pd.read_parquet(METADATA_PATH)

if __name__ == "__main__":
    generate_embeddings(batch_size=32)
