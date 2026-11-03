
import os
import urllib.request
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

INPUT_TXT = "newspaper_pages_links.txt"
DOWNLOAD_FOLDER = "downloaded_newspapers"

MAX_WORKERS = 10
MAX_RETRIES = 3
TIMEOUT = 30


def download_image(index, total, file_name, url):
    """Download one image, retrying if necessary."""

    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    temp_path = file_path + ".tmp"

    # Skip only if a non-empty file already exists
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return index, "skipped", file_name

    # Remove an empty or incomplete existing file
    if os.path.exists(file_path):
        os.remove(file_path)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            with urllib.request.urlopen(
                request, timeout=TIMEOUT
            ) as response:
                with open(temp_path, "wb") as output:
                    shutil.copyfileobj(response, output)

            # Verify that the downloaded file is not empty
            if os.path.getsize(temp_path) == 0:
                raise ValueError("Downloaded file is empty.")

            # Move completed download into its final location
            os.replace(temp_path, file_path)

            return index, "downloaded", file_name

        except Exception as error:
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if attempt == MAX_RETRIES:
                return index, "failed", f"{file_name}: {error}"

            time.sleep(1)

    return index, "failed", file_name


def main():

    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    if not os.path.exists(INPUT_TXT):
        print(f"Error: File '{INPUT_TXT}' not found.")
        return

    # Read and prepare download tasks
    tasks = []
    downloaded_urls = set()

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for index, line in enumerate(lines, 1):
        line = line.strip()

        if not line or " | " not in line:
            continue

        file_name, url = line.split(" | ", 1)

        if url in downloaded_urls:
            continue

        downloaded_urls.add(url)
        tasks.append((index, len(lines), file_name, url))

    total = len(tasks)

    print(f"Total unique images: {total}")
    print(f"Concurrent downloads: {MAX_WORKERS}")
    print("\nStarting downloads...\n")

    downloaded = 0
    skipped = 0
    failed = 0

    start_time = time.time()

    # Download multiple images concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(download_image, *task)
            for task in tasks
        ]

        for completed, future in enumerate(as_completed(futures), 1):

            index, status, result = future.result()

            if status == "downloaded":
                downloaded += 1
                print(f"[{completed}/{total}] Downloaded: {result}")

            elif status == "skipped":
                skipped += 1
                print(f"[{completed}/{total}] Already exists: {result}")

            else:
                failed += 1
                print(f"[{completed}/{total}] FAILED: {result}")

    elapsed = time.time() - start_time

    print("\n========== DOWNLOAD SUMMARY ==========")
    print("Total unique URLs:", total)
    print("New downloads:", downloaded)
    print("Already existed:", skipped)
    print("Failed downloads:", failed)
    print("Elapsed time:", round(elapsed / 60, 2), "minutes")
    print("======================================")


if __name__ == "__main__":
    main()
