import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import os
import sys

# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA = Path(__file__).resolve().parent.parent

MAPPING_FILE = DATA / "newspaper_pages_with_publisher_links.txt"
ORIGINAL_LINKS_FILE = DATA / "newspaper_pages_links.txt"
METADATA_DIR = DATA / "dataset" / "metadata" / "per_page"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

BACKUP_DIR = (
    DATA / "dataset" / "metadata" / f"per_page_backup_{TIMESTAMP}"
)

AUDIT_FILE = DATA / f"publisher_metadata_audit_{TIMESTAMP}.csv"

# Fields this script will add to each crop record
NEW_FIELDS = {
    "publisher",
    "lccn",
    "publisher_candidates",
    "publisher_match_status",
}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def extract_lccn(original_filename, unique_filename):
    """Extract LCCN from the unique filename."""

    prefix = original_filename.rsplit("_page_", 1)[0]
    unique_prefix = unique_filename.rsplit("_page_", 1)[0]

    expected_prefix = prefix + "_"

    if not unique_prefix.startswith(expected_prefix):
        raise ValueError(
            f"Cannot extract LCCN from {unique_filename}"
        )

    return unique_prefix[len(expected_prefix):]


def read_mapping():
    """Read URL -> publisher/LCCN mapping."""

    map_by_url = {}

    with open(MAPPING_FILE, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            parts = [part.strip() for part in line.split("|", 3)]

            if len(parts) != 4:
                raise ValueError(
                    f"Invalid mapping line {line_number}: {line}"
                )

            original_filename, unique_filename, publisher, url = parts

            lccn = extract_lccn(
                original_filename,
                unique_filename
            )

            candidate = {
                "publisher": publisher,
                "lccn": lccn,
            }

            if url in map_by_url:
                if map_by_url[url] != candidate:
                    raise ValueError(
                        f"Conflicting publisher mappings for URL: {url}"
                    )
            else:
                map_by_url[url] = candidate

    return map_by_url


def read_ordered_candidates(map_by_url):
    """
    Read the ORIGINAL link file sequentially.
    This preserves the original appearance order of candidates.
    """

    candidates_by_filename = defaultdict(list)
    missing_urls = []

    with open(ORIGINAL_LINKS_FILE, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            if " | " not in line:
                raise ValueError(
                    f"Invalid original link line {line_number}: {line}"
                )

            filename, url = line.split(" | ", 1)

            candidate = map_by_url.get(url)

            if candidate is None:
                missing_urls.append((filename, url))
                continue

            # Deduplicate while preserving first appearance.
            identity = (
                candidate["publisher"],
                candidate["lccn"],
            )

            existing = {
                (c["publisher"], c["lccn"])
                for c in candidates_by_filename[filename]
            }

            if identity not in existing:
                candidates_by_filename[filename].append(candidate)

    return candidates_by_filename, missing_urls


def get_assignment(candidates):
    """Build the agreed publisher metadata structure."""

    if not candidates:
        return {
            "publisher": None,
            "lccn": None,
            "publisher_candidates": [],
            "publisher_match_status": "unresolved",
        }

    first = candidates[0]

    if len(candidates) == 1:
        return {
            "publisher": first["publisher"],
            "lccn": first["lccn"],
            "publisher_match_status": "single_candidate",
        }

    return {
        "publisher": first["publisher"],
        "lccn": first["lccn"],
        "publisher_candidates": [
            dict(candidate) for candidate in candidates[1:]
        ],
        "publisher_match_status": "ambiguous",
    }


# --------------------------------------------------
# LOAD MAPPING AND ORIGINAL LINK ORDER
# --------------------------------------------------

print("Loading publisher mapping...")

map_by_url = read_mapping()

print("Mapping URLs:", len(map_by_url))

candidates_by_filename, missing_urls = read_ordered_candidates(
    map_by_url
)

print(
    "Original filenames with publisher candidates:",
    len(candidates_by_filename)
)

print("Original URLs without a mapping:", len(missing_urls))


# --------------------------------------------------
# INSPECT ALL METADATA FILES
# --------------------------------------------------

metadata_files = sorted(METADATA_DIR.glob("*.json"))

print("Metadata JSON files:", len(metadata_files))

plans = []
invalid_files = []
empty_files = []
nonempty_files = []
structural_errors = []

for path in metadata_files:

    inferred_source = path.stem + ".jpg"

    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

    except Exception as e:
        # Invalid JSON: skip it, but retain it in the audit.
        invalid_files.append((path, str(e)))

        candidates = candidates_by_filename.get(
            inferred_source, []
        )

        plans.append({
            "path": path,
            "source_filename": inferred_source,
            "records": None,
            "candidates": candidates,
            "assignment": get_assignment(candidates),
            "metadata_status": "invalid_json_skipped",
        })

        continue

    if not isinstance(records, list):
        structural_errors.append(
            (path.name, "Top-level JSON value is not a list")
        )
        continue

    # Empty list is valid. Do not modify it.
    if len(records) == 0:
        empty_files.append(path.name)

        candidates = candidates_by_filename.get(
            inferred_source, []
        )

        plans.append({
            "path": path,
            "source_filename": inferred_source,
            "records": [],
            "candidates": candidates,
            "assignment": get_assignment(candidates),
            "metadata_status": "empty_no_crops",
        })

        continue

    nonempty_files.append(path.name)

    # Validate crop records.
    source_names = set()

    for record in records:
        if not isinstance(record, dict):
            structural_errors.append(
                (path.name, "A crop record is not a dictionary")
            )
            break

        if "source_filename" not in record:
            structural_errors.append(
                (path.name, "A crop record has no source_filename")
            )
            break

        source_names.add(record["source_filename"])

    if any(name[0] == path.name for name in structural_errors):
        continue

    if len(source_names) != 1:
        structural_errors.append(
            (path.name, f"Inconsistent source filenames: {source_names}")
        )
        continue

    source_filename = next(iter(source_names))

    candidates = candidates_by_filename.get(
        source_filename, []
    )

    # Do not overwrite fields if they already exist.
    existing_fields = NEW_FIELDS.intersection(
        set().union(*(record.keys() for record in records))
    )

    if existing_fields:
        structural_errors.append(
            (
                path.name,
                f"Publisher fields already exist: {existing_fields}"
            )
        )
        continue

    plans.append({
        "path": path,
        "source_filename": source_filename,
        "records": records,
        "candidates": candidates,
        "assignment": get_assignment(candidates),
        "metadata_status": "ready_to_update",
    })


# --------------------------------------------------
# PREFLIGHT SUMMARY
# --------------------------------------------------

print("\n========== PREFLIGHT ==========")

print("Total metadata files:", len(metadata_files))
print("Non-empty valid files:", len(nonempty_files))
print("Empty valid files:", len(empty_files))
print("Invalid JSON files to skip:", len(invalid_files))
print("Structural errors:", len(structural_errors))

if structural_errors:
    print("\nPREFLIGHT FAILED. No files have been modified.")

    for filename, error in structural_errors[:30]:
        print(filename, ":", error)

    print("\nPlease resolve these structural issues first.")
    sys.exit(1)

if len(plans) != len(metadata_files):
    print("\nPREFLIGHT FAILED. Some files were not accounted for.")
    sys.exit(1)

single_count = sum(
    p["assignment"]["publisher_match_status"] == "single_candidate"
    for p in plans
)

ambiguous_count = sum(
    p["assignment"]["publisher_match_status"] == "ambiguous"
    for p in plans
)

unresolved_count = sum(
    p["assignment"]["publisher_match_status"] == "unresolved"
    for p in plans
)

print("\nPublisher assignment summary:")
print("Single candidate:", single_count)
print("Ambiguous:", ambiguous_count)
print("Unresolved:", unresolved_count)

print("\nInvalid files will be skipped, not deleted or overwritten.")
print("Empty files will remain exactly as they are.")
print("Original images, crops, TXT files and filenames will not be changed.")

print("\nBackup location:")
print(BACKUP_DIR)

print("\nAudit report:")
print(AUDIT_FILE)

confirmation = input(
    "\nType APPLY to create the backup and update metadata: "
).strip()

if confirmation != "APPLY":
    print("\nCancelled. No files have been modified.")
    sys.exit(0)


# --------------------------------------------------
# CREATE COMPLETE BACKUP
# --------------------------------------------------

print("\nCreating complete backup...")

BACKUP_DIR.mkdir(parents=True, exist_ok=False)

try:
    for path in metadata_files:
        shutil.copy2(path, BACKUP_DIR / path.name)

    backup_count = len(list(BACKUP_DIR.glob("*.json")))

    if backup_count != len(metadata_files):
        raise RuntimeError(
            f"Backup incomplete: {backup_count}/{len(metadata_files)}"
        )

    print("Backup completed:", backup_count, "files")


    # --------------------------------------------------
    # UPDATE NON-EMPTY VALID JSON FILES
    # --------------------------------------------------

    updated_count = 0
    unchanged_empty_count = 0
    skipped_invalid_count = 0

    for plan in plans:

        path = plan["path"]
        records = plan["records"]
        assignment = plan["assignment"]

        if records is None:
            skipped_invalid_count += 1
            continue

        if len(records) == 0:
            unchanged_empty_count += 1
            continue

        for record in records:
            for key, value in assignment.items():
                record[key] = value

        temp_path = path.with_suffix(".json.tmp")

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                records,
                f,
                ensure_ascii=False,
                indent=2
            )
            f.write("\n")

        os.replace(temp_path, path)

        updated_count += 1


    # --------------------------------------------------
    # WRITE PAGE-LEVEL AUDIT CSV
    # --------------------------------------------------

    print("\nWriting audit report...")

    with open(AUDIT_FILE, "w", newline="", encoding="utf-8-sig") as f:

        fieldnames = [
            "metadata_file",
            "source_filename",
            "crop_count",
            "publisher",
            "lccn",
            "publisher_candidates",
            "publisher_candidate_count",
            "publisher_match_status",
            "metadata_status",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for plan in plans:

            assignment = plan["assignment"]
            records = plan["records"]

            candidates = plan["candidates"]

            writer.writerow({
                "metadata_file": plan["path"].name,
                "source_filename": plan["source_filename"],
                "crop_count": (
                    len(records) if records is not None else ""
                ),
                "publisher": assignment.get("publisher"),
                "lccn": assignment.get("lccn"),
                "publisher_candidates": json.dumps(
                    assignment.get("publisher_candidates", []),
                    ensure_ascii=False
                ),
                "publisher_candidate_count": len(candidates),
                "publisher_match_status": assignment[
                    "publisher_match_status"
                ],
                "metadata_status": plan["metadata_status"],
            })


    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n========== COMPLETED ==========")

    print("Metadata files updated:", updated_count)
    print("Empty files left unchanged:", unchanged_empty_count)
    print("Invalid JSON files skipped:", skipped_invalid_count)
    print("Audit rows written:", len(plans))
    print("Backup files:", backup_count)

    print("\nBackup:", BACKUP_DIR)
    print("Audit:", AUDIT_FILE)

    print(
        "\nIMPORTANT: publisher is the first candidate "
        "in the original link-file order. "
        "For ambiguous pages, it is not a verified publisher identity."
    )

except Exception as e:

    print("\nERROR DURING UPDATE:", e)
    print("Restoring all metadata JSON files from backup...")

    for backup_path in BACKUP_DIR.glob("*.json"):
        original_path = METADATA_DIR / backup_path.name
        shutil.copy2(backup_path, original_path)

    for temp_path in METADATA_DIR.glob("*.json.tmp"):
        temp_path.unlink(missing_ok=True)

    print("Restoration completed.")
    print("Please do not rerun until the error has been reviewed.")
    sys.exit(1)