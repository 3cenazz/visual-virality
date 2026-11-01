import json
import os
import re

json_folder = "json_files"
output_txt = "newspaper_pages_links.txt"

if not os.path.exists(json_folder):
    print(f"Error: Directory '{json_folder}' not found. Please create it and put your JSON files inside.")
else:
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in '{json_folder}' directory.")
    else:
        total_link_count = 0
        unique_link_count = 0
        
        # Using a set for ultra-fast O(1) duplicate checking without slowing down performance
        seen_links = set()

        print(f"Found {len(json_files)} JSON file(s). Processing and filtering duplicates...\n")

        with open(output_txt, "w", encoding="utf-8") as txt_file:
            for json_file_name in json_files:
                file_path = os.path.join(json_folder, json_file_name)
                print(f"Processing file: {json_file_name}...")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    results = data.get("content", {}).get("results", [])
                    file_unique_count = 0

                    for item in results:
                        date_str = item.get("date", "unknown_date")
                        resources = item.get("resources", [])

                        for res in resources:
                            file_count = res.get("files", 0)
                            first_image_url = res.get("image", "")

                            if first_image_url and file_count > 0:
                                match = re.search(r"(\d{4})/full/", first_image_url)
                                if match:
                                    start_id_str = match.group(1)
                                    start_id = int(start_id_str)
                                    base_url = first_image_url.split(start_id_str + "/full/")[0]

                                    for i in range(file_count):
                                        current_id = f"{start_id + i:04d}"
                                        full_image_url = f"{base_url}{current_id}/full/full/0/default.jpg"
                                        
                                        # Check if this URL has already been processed
                                        if full_image_url not in seen_links:
                                            seen_links.add(full_image_url)
                                            
                                            file_name_str = f"{date_str}_page_{i+1:02d}.jpg"
                                            txt_file.write(f"{file_name_str} | {full_image_url}\n")
                                            
                                            file_unique_count += 1
                                            unique_link_count += 1
                                        
                                        total_link_count += 1

                    print(f" -> Added {file_unique_count} unique links from {json_file_name}.")

                except json.JSONDecodeError:
                    print(f" -> Error: Invalid JSON format in '{json_file_name}', skipped.")
                except Exception as e:
                    print(f" -> Error processing '{json_file_name}': {e}")

        duplicate_count = total_link_count - unique_link_count
        print(f"\n=== PROCESS COMPLETED ===")
        print(f"Total scanned links: {total_link_count}")
        print(f"Duplicates removed: {duplicate_count}")
        print(f"Unique links saved to '{output_txt}': {unique_link_count}")