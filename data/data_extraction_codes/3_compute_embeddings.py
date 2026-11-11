"""
Compute image embeddings for your group's Newspaper Navigator 1956 copy.

Input, either
  (a) a single metadata JSON (list of entries with "crop_filename", "publication_date", ...)
      + the folder that contains the crop images        -> --meta ... --crops-dir ...
  (b) the shared dataset folder (e.g. the OneDrive/SharePoint sync), which contains
      images/ and metadata/per_page/*.json              -> --dataset ...

Output (in --out-dir):
  embeddings_<model>.npy       one row per image, L2-normalised
  embeddings_<model>_meta.csv  same row order: crop_filename, date, category, ...
  metadata_all.json            (only with --dataset) all per-page metadata merged
  chunks_<model>/              checkpoints; re-running the same command resumes

Works with OneDrive "Files On-Demand": images are fetched in parallel, download
timeouts are retried, and on macOS files that were cloud-only before are evicted
again right after being embedded, so the full dataset never has to fit on disk.

Usage:
  pip install torch torchvision pillow pandas numpy tqdm
  python 3_compute_embeddings.py --meta metadata.json --crops-dir crops/

  # straight from the shared dataset folder, skipping advertisements:
  python 3_compute_embeddings.py \
      --dataset "~/Library/CloudStorage/OneDrive-epfl.ch/Dateien von Ecenaz Egri - dataset" \
      --exclude Advertisement

  # optional, CLIP instead of ResNet-18 (pip install transformers):
  python 3_compute_embeddings.py --meta metadata.json --crops-dir crops/ --model clip

  # keep only confident detections:
  python 3_compute_embeddings.py --meta metadata.json --crops-dir crops/ --min-confidence 0.5
"""

import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

SF_DATALESS = 0x40000000  # macOS st_flags bit for cloud-only placeholder files

EVICT_SWIFT = r"""
import Foundation
var fails = 0
for p in CommandLine.arguments.dropFirst() {
  do { try FileManager.default.evictUbiquitousItem(at: URL(fileURLWithPath: p)) }
  catch { fails += 1 }
}
exit(fails == 0 ? 0 : 1)
"""


# ---------------------------------------------------------------- model
def load_model(name, device):
    if name == "resnet18":
        import torchvision.models as tvm
        import torchvision.transforms as T

        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()          # 512-dim features (penultimate layer)
        m.eval().to(device)
        tf = T.Compose([
            T.Resize(256), T.CenterCrop(224), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return tf, lambda x: m(x)

    if name == "clip":
        from transformers import CLIPModel, CLIPProcessor

        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        tf = lambda img: proc(images=img, return_tensors="pt")["pixel_values"][0]
        return tf, lambda x: m.get_image_features(pixel_values=x)

    raise ValueError(f"unknown model {name}")


# ---------------------------------------------------------------- file helpers
def read_bytes(path, attempts=6):
    """Read a file, retrying (OneDrive) download timeouts with backoff (5, 10, 20, 40, 80 s)."""
    for k in range(attempts):
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            if k == attempts - 1 or not os.path.exists(path):
                raise
            time.sleep(5 * 2 ** k)


def build_evict_helper(out_dir):
    """Compile a tiny Swift tool that turns downloaded OneDrive files back into cloud-only ones (macOS)."""
    if sys.platform != "darwin" or not shutil.which("swiftc"):
        return None
    exe = os.path.join(out_dir, ".evict")
    if not os.path.exists(exe):
        src = exe + ".swift"
        with open(src, "w") as f:
            f.write(EVICT_SWIFT)
        if subprocess.run(["swiftc", "-O", src, "-o", exe], capture_output=True).returncode != 0:
            print("Could not build evict helper; downloaded files will stay on disk.")
            return None
    return exe


def load_metadata(a):
    if a.meta:
        return json.loads(read_bytes(a.meta))

    cache = os.path.join(a.out_dir, "metadata_all.json")
    if os.path.exists(cache):
        return json.loads(read_bytes(cache))
    files = sorted(glob.glob(os.path.join(a.dataset, "metadata", "per_page", "*.json")))
    rows = []
    with ThreadPoolExecutor(64) as pool:  # per-page files can be cloud-only too
        for r in tqdm(pool.map(lambda fn: json.loads(read_bytes(fn)), files), total=len(files), desc="metadata"):
            rows.extend(r)
    with open(cache, "w") as f:
        json.dump(rows, f)
    return rows


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--meta", help="metadata JSON file (use with --crops-dir)")
    src.add_argument("--dataset", help="dataset root containing images/ and metadata/per_page/")
    p.add_argument("--crops-dir", help="folder that crop_filename paths are relative to "
                                       "(default with --dataset: <dataset>/images)")
    p.add_argument("--model", choices=["resnet18", "clip"], default="resnet18")
    p.add_argument("--out-dir", default="embeddings_out")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--chunk-size", type=int, default=512, help="images per checkpoint")
    p.add_argument("--workers", type=int, default=32, help="parallel file reads/downloads")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--exclude", nargs="*", default=[], help="categories to skip, e.g. Advertisement")
    p.add_argument("--keep-downloaded", action="store_true",
                   help="don't turn OneDrive files back into cloud-only after embedding")
    a = p.parse_args()

    if a.dataset:
        a.dataset = os.path.expanduser(a.dataset)
        a.crops_dir = a.crops_dir or os.path.join(a.dataset, "images")
    elif not a.crops_dir:
        p.error("--crops-dir is required with --meta")
    os.makedirs(a.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(a.out_dir, f"chunks_{a.model}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- metadata
    meta = pd.DataFrame(load_metadata(a))
    meta = meta[meta["confidence"] >= a.min_confidence]
    if a.exclude:
        meta = meta[~meta["category"].isin(a.exclude)]
    meta = meta.sort_values("publication_date", kind="stable").reset_index(drop=True)
    print(f"{len(meta):,} images after filtering")
    print(meta["category"].value_counts().to_string())

    # ---- model
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    tf, encode = load_model(a.model, device)
    evict = None if a.keep_downloaded else build_evict_helper(a.out_dir)

    def load(i):
        path = os.path.join(a.crops_dir, meta.at[i, "crop_filename"])
        try:
            was_cloud = bool(os.stat(path).st_flags & SF_DATALESS)
            img = Image.open(io.BytesIO(read_bytes(path))).convert("RGB")  # read triggers the download
            return i, tf(img), (path if was_cloud else None)
        except Exception:
            return i, None, None

    def embed_chunk(pool, s, e):
        vecs, idx, batch, bidx, to_evict = [], [], [], [], []

        def flush():
            with torch.no_grad():
                v = encode(torch.stack(batch).to(device)).float().cpu().numpy()
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            vecs.append(v)
            idx.extend(bidx)
            batch.clear()
            bidx.clear()

        for i, t, cloud_path in pool.map(load, range(s, e)):
            pbar.update(1)
            if t is None:
                continue
            batch.append(t)
            bidx.append(i)
            if cloud_path:
                to_evict.append(cloud_path)
            if len(batch) == a.batch_size:
                flush()
        if batch:
            flush()
        emb = np.concatenate(vecs) if vecs else np.zeros((0, 0), np.float32)
        return emb, idx, to_evict

    # ---- embed in checkpointed chunks
    n = len(meta)
    chunks = [(s, min(s + a.chunk_size, n)) for s in range(0, n, a.chunk_size)]
    pbar = tqdm(total=n, desc="embedding")
    with ThreadPoolExecutor(a.workers) as pool:
        for s, e in chunks:
            ck = os.path.join(ckpt_dir, f"{s:07d}.npz")
            if os.path.exists(ck):
                pbar.update(e - s)
                continue
            emb, idx, to_evict = embed_chunk(pool, s, e)
            np.savez(ck + ".tmp.npz", emb=emb, idx=np.array(idx, dtype=np.int64))
            os.replace(ck + ".tmp.npz", ck)
            if evict and to_evict:
                subprocess.run([evict, *to_evict], capture_output=True)
    pbar.close()

    # ---- save
    embs, kept = [], []
    for s, _ in chunks:
        d = np.load(os.path.join(ckpt_dir, f"{s:07d}.npz"))
        if len(d["idx"]):
            embs.append(d["emb"])
            kept.extend(d["idx"].tolist())
    emb = np.concatenate(embs)
    np.save(os.path.join(a.out_dir, f"embeddings_{a.model}.npy"), emb)
    meta.iloc[kept].to_csv(os.path.join(a.out_dir, f"embeddings_{a.model}_meta.csv"), index=False)

    print(f"\nSaved {emb.shape[0]:,} x {emb.shape[1]} embeddings to {a.out_dir}/")
    if len(kept) < n:
        print(f"{n - len(kept):,} images could not be opened (missing or broken files)")


if __name__ == "__main__":
    main()
