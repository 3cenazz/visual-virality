import os
import urllib.request

input_txt = "newspaper_pages_links.txt"
download_folder = "downloaded_newspapers"

if not os.path.exists(download_folder):
    os.makedirs(download_folder)

if not os.path.exists(input_txt):
    print(f"Error: File '{input_txt}' not found. Please run the first script first.")
else:
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Starting to download images from '{input_txt}'...\n")

    # Set to prevent duplicate downloads in the current session
    downloaded_urls = set()

    for index, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        if " | " in line:
            file_name_str, url = line.split(" | ", 1)
            
            # Skip if already processed in this session
            if url in downloaded_urls:
                continue
            downloaded_urls.add(url)

            file_path = os.path.join(download_folder, file_name_str)

            # Skip if already exists on disk
            if os.path.exists(file_path):
                print(f"[{index}/{total_lines}] Already exists, skipped: {file_name_str}")
                continue

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as response, open(file_path, "wb") as out_file:
                    out_file.write(response.read())
                
                print(f"[{index}/{total_lines}] Downloaded: {file_name_str}")
            except Exception as e:
                print(f"[{index}/{total_lines}] Error ({file_name_str}): {e}")

    print("\n=== ALL DOWNLOADS COMPLETED ===")