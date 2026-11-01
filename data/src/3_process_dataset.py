
"""
Newspaper Navigator dataset processing pipeline.

1. Read each original newspaper page.
2. Run object detection.
3. Crop detected objects from the original full-resolution image.
4. Save crops directly into category-specific folders.
5. Exclude the Headline class.
6. Record metadata for every saved crop.
7. Mark successfully processed pages so processing can resume.
"""

import argparse
import json
import shutil
import time

import cv2
import torch

from src.config import (
    SOURCE_DIR,
    CROPS_DIR,
    METADATA_DIR,
    COMPLETED_DIR,
    CLASS_NAMES,
)

from src.model import load_model
from src.metadata import create_crop_metadata


# Category folder names used in the dataset.
CATEGORY_FOLDERS = {
    "Illustration/Photograph": "illustration_photograph",
    "Photograph": "photograph",
    "Comics/Cartoon": "comics_cartoon",
    "Editorial Cartoon": "editorial_cartoon",
    "Map": "map",
    "Advertisement": "advertisement",
}


# Headline detections are deliberately excluded.
EXCLUDED_CLASSES = {"Headline"}


def get_page_files():
    """Return all newspaper page images in sorted order."""

    return sorted(
        f for f in SOURCE_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg")
    )


def clean_previous_page_outputs(page_name):
    """
    Remove crops and metadata from an incomplete previous attempt.

    Since crops are now stored directly in category folders,
    remove files belonging to this page from each category folder.
    """

    for category_folder in CATEGORY_FOLDERS.values():

        category_dir = CROPS_DIR / category_folder

        if category_dir.exists():

            for crop_path in category_dir.glob(f"{page_name}_*.jpg"):
                crop_path.unlink()

    metadata_path = METADATA_DIR / f"{page_name}.json"

    if metadata_path.exists():
        metadata_path.unlink()


def process_page(predictor, filename):
    """Detect objects and save their crops and metadata for one page."""

    source_path = SOURCE_DIR / filename
    page_name = source_path.stem

    metadata_path = METADATA_DIR / f"{page_name}.json"
    completed_marker = COMPLETED_DIR / f"{page_name}.done"

    # Skip pages that have already been completed.
    if completed_marker.exists():
        print(f"Skipping completed page: {filename}")
        return "skipped", 0

    # Remove incomplete output from an earlier interrupted attempt.
    clean_previous_page_outputs(page_name)

    # Check whether the source image is empty.
    if not source_path.exists() or source_path.stat().st_size == 0:

        print(f"WARNING: Empty or missing image. Skipping: {filename}")

        if source_path.exists():
            source_path.unlink()
            print(f"Deleted invalid source file: {filename}")

        return "invalid", 0

    # Attempt to read the image.
    image = cv2.imread(str(source_path))

    if image is None:

        print(f"WARNING: Unreadable image. Skipping: {filename}")

        source_path.unlink()
        print(f"Deleted invalid source file: {filename}")

        return "invalid", 0

    height, width = image.shape[:2]

    # Run Newspaper Navigator detection.
    with torch.no_grad():
        outputs = predictor(image)

    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()

    page_metadata = []
    saved_count = 0
    excluded_count = 0

    for i, (box, score, class_id) in enumerate(
        zip(boxes, scores, classes)
    ):

        category = CLASS_NAMES[int(class_id)]

        # Do not save headline detections.
        if category in EXCLUDED_CLASSES:
            excluded_count += 1
            continue

        # Map the detected class to its dataset folder.
        category_folder = CATEGORY_FOLDERS.get(category)

        if category_folder is None:
            print(f"WARNING: Unknown category: {category}")
            continue

        # Convert detection coordinates to integers.
        x1, y1, x2, y2 = map(int, box)

        # Keep coordinates within original image boundaries.
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop from the original full-resolution page.
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Save directly into the category folder.
        category_dir = CROPS_DIR / category_folder
        category_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = (
            f"{page_name}_{category_folder}_{i:03d}.jpg"
        )

        crop_path = category_dir / crop_filename

        if not cv2.imwrite(str(crop_path), crop):
            raise IOError(f"Could not save crop: {crop_path}")

        # Record metadata for this crop.
        record = create_crop_metadata(
            source_filename=source_path.name,
            crop_filename=str(crop_path.relative_to(CROPS_DIR)),
            category=category,
            confidence=score,
            box=(x1, y1, x2, y2),
        )

        page_metadata.append(record)
        saved_count += 1

    # Save metadata for this page.
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(page_metadata, f, indent=2)

    # Only mark the page complete after all outputs are saved.
    completed_marker.parent.mkdir(parents=True, exist_ok=True)

    completed_marker.write_text(
        "completed\n",
        encoding="utf-8"
    )

    print(
        f"Completed {filename}: "
        f"{len(instances)} detections, "
        f"{excluded_count} headlines excluded, "
        f"{saved_count} crops saved."
    )

    return "processed", saved_count


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of unfinished pages to process.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    # Verify input data.
    files = get_page_files()

    print("Newspaper pages found:", len(files))

    # Select only pages that have not been completed yet.
    pending_files = [
        filename for filename in files
        if not (
            COMPLETED_DIR / f"{filename.stem}.done"
        ).exists()
    ]

    print("Pages remaining:", len(pending_files))

    # Limit the number of pages processed in this run.
    if args.limit is not None:
        pending_files = pending_files[:args.limit]

    print("Pages selected for this run:", len(pending_files))

    if not pending_files:
        print("No unfinished pages to process.")
        return

    # Create the six category directories.
    for category_folder in CATEGORY_FOLDERS.values():
        (CROPS_DIR / category_folder).mkdir(
            parents=True,
            exist_ok=True
        )

    # Load model once, not once per page.
    predictor = load_model()

    total_crops = 0
    processed_pages = 0
    invalid_pages = 0

    start_time = time.time()

    for index, filename in enumerate(pending_files, start=1):

        print(
            f"\n[{index}/{len(pending_files)}] "
            f"Processing {filename}"
        )

        status, crop_count = process_page(
            predictor,
            filename
        )

        if status == "processed":
            processed_pages += 1
            total_crops += crop_count

        elif status == "invalid":
            invalid_pages += 1

    elapsed = time.time() - start_time

    print("\nDataset processing finished!")
    print("New pages processed:", processed_pages)
    print("Invalid pages deleted:", invalid_pages)
    print("Total crops saved:", total_crops)
    print("Elapsed time:", round(elapsed / 60, 2), "minutes")


if __name__ == "__main__":
    main()
