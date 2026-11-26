from pathlib import Path
from collections import defaultdict
from urllib.request import Request, urlopen
from tempfile import TemporaryDirectory
from PIL import Image, ImageOps, ImageChops
import random
import csv
import io
import re
import time

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MAPPING_FILE = Path("newspaper_pages_with_publisher_links.txt")
ORIGINAL_LINKS_FILE = Path("newspaper_pages_links.txt")
REPORT_FILE = Path("first_occurrence_validation.csv")
IMAGE_FOLDER = Path("downloaded_newspapers")

SAMPLE_SIZE = 15
RANDOM_SEED = 42
DOWNLOAD_RETRIES = 3

# --------------------------------------------------
# 1. READ PUBLISHER MAPPING
# --------------------------------------------------

url_info = {}

with open(MAPPING_FILE, encoding="utf-8") as f:
    for line in f:
        parts = [p.strip() for p in line.strip().split("|")]

        if len(parts) != 4:
            continue

        original, unique, publisher, url = parts

        match = re.match(
            r"^\d{4}-\d{2}-\d{2}_(.+)_page_\d+\.jpg$",
            unique
        )

        lccn = match.group(1) if match else ""

        url_info[url] = {
            "publisher": publisher,
            "lccn": lccn,
            "unique_filename": unique
        }

# --------------------------------------------------
# 2. READ FIRST OCCURRENCES
# --------------------------------------------------

first_occurrences = {}

with open(ORIGINAL_LINKS_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line or " | " not in line:
            continue

        filename, url = line.split(" | ", 1)

        if filename not in first_occurrences:
            first_occurrences[filename] = url

# --------------------------------------------------
# 3. IDENTIFY AMBIGUOUS LOCAL IMAGES
# --------------------------------------------------

original_candidates = defaultdict(set)

with open(MAPPING_FILE, encoding="utf-8") as f:
    for line in f:
        parts = [p.strip() for p in line.strip().split("|")]

        if len(parts) != 4:
            continue

        original, unique, publisher, url = parts
        original_candidates[original].add(url)

if not IMAGE_FOLDER.is_dir():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_FOLDER}")

local_images = sorted(
    p.name for p in IMAGE_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
)

ambiguous_images = [
    name for name in local_images
    if len(original_candidates[name]) > 1
    and name in first_occurrences
]

# --------------------------------------------------
# 4. SELECT 15 IMAGES FROM DIFFERENT DATES
# --------------------------------------------------

random.seed(RANDOM_SEED)
random.shuffle(ambiguous_images)

selected = []
used_dates = set()

for filename in ambiguous_images:
    date = filename[:10]

    if date not in used_dates:
        selected.append(filename)
        used_dates.add(date)

    if len(selected) == SAMPLE_SIZE:
        break

if len(selected) < SAMPLE_SIZE:
    for filename in ambiguous_images:
        if filename not in selected:
            selected.append(filename)

        if len(selected) == SAMPLE_SIZE:
            break

print(f"Selected images: {len(selected)}")

# --------------------------------------------------
# 5. IMAGE COMPARISON
# --------------------------------------------------

def compare_images(local_path, candidate_bytes):

    with Image.open(local_path) as img:
        local = ImageOps.grayscale(img)
        local = ImageOps.pad(
            local, (256, 256), color=255
        )

    with Image.open(io.BytesIO(candidate_bytes)) as img:
        candidate = ImageOps.grayscale(img)
        candidate = ImageOps.pad(
            candidate, (256, 256), color=255
        )

    difference = ImageChops.difference(local, candidate)

    return sum(difference.getdata()) / (256 * 256)


# --------------------------------------------------
# 6. DOWNLOAD AND VALIDATE
# --------------------------------------------------

results = []

with TemporaryDirectory(prefix="newspaper_validation_") as temp_dir:

    for index, filename in enumerate(selected, 1):

        local_path = IMAGE_FOLDER / filename
        url = first_occurrences[filename]
        info = url_info.get(url)

        print(f"\n[{index}/{len(selected)}] {filename}")

        result = {
            "original_filename": filename,
            "publisher_candidate": info["publisher"] if info else "",
            "lccn_candidate": info["lccn"] if info else "",
            "unique_filename": info["unique_filename"] if info else "",
            "similarity_difference": "",
            "result": ""
        }

        if not info:
            result["result"] = "MAPPING_NOT_FOUND"
            results.append(result)
            continue

        candidate_bytes = None

        for attempt in range(1, DOWNLOAD_RETRIES + 1):

            try:
                request = Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )

                with urlopen(request, timeout=120) as response:
                    candidate_bytes = response.read()

                break

            except Exception as error:
                print(f"  Download attempt {attempt} failed: {error}")

                if attempt < DOWNLOAD_RETRIES:
                    time.sleep(3)

        if candidate_bytes is None:
            result["result"] = "DOWNLOAD_FAILED"
            results.append(result)
            continue

        try:
            difference = compare_images(
                local_path,
                candidate_bytes
            )

            result["similarity_difference"] = round(difference, 3)
            result["result"] = "COMPARED"

            print("  Publisher:", info["publisher"])
            print("  LCCN:", info["lccn"])
            print("  Mean pixel difference:", round(difference, 3))

        except Exception as error:
            result["result"] = "COMPARISON_FAILED"
            print("  Comparison error:", repr(error))

        results.append(result)

# --------------------------------------------------
# 7. SAVE REPORT
# --------------------------------------------------

if results:
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

print("\n========== VALIDATION SUMMARY ==========")
print("Images selected:", len(results))
print("Successfully compared:", sum(
    r["result"] == "COMPARED" for r in results
))
print("Download/comparison errors:", sum(
    r["result"] != "COMPARED" for r in results
))
print("Report:", REPORT_FILE)
print("Temporary downloads automatically removed.")
print("Original dataset and metadata were not modified.")
print("========================================")
