# Newspaper Image Dataset: Data Extraction Pipeline

This repository contains the data extraction pipeline for the Visual Virality project. The pipeline collects historical newspaper page images from the Library of Congress's Chronicling America collection and uses the Newspaper Navigator object detection model to extract visual elements from those pages.

The resulting dataset consists of categorized image crops and their corresponding metadata.

## 1. Pipeline Overview

The data extraction process consists of the following stages:

1. **Source data collection:** Download JSON records from the Library of Congress Chronicling America collection.
2. **Image URL extraction:** Extract and consolidate newspaper page image URLs from the JSON files.
3. **Image downloading:** Download the original newspaper page images locally using a multithreaded downloader.
4. **Object detection:** Apply the Newspaper Navigator object detection model to identify visual elements on each newspaper page.
5. **Image cropping:** Extract detected objects from the original, full-resolution newspaper pages.
6. **Metadata generation:** Save information about each extracted crop, including its source page, category, confidence score, and bounding box.

The pipeline is designed to process multiple newspaper pages and resume processing without repeating pages that have already been completed.

## 2. Data Source

The project uses historical newspaper page images from **Chronicling America**, provided by the Library of Congress.

- Chronicling America: https://www.loc.gov/collections/chronicling-america/
- Library of Congress: https://www.loc.gov/
- Newspaper Navigator: https://news-navigator.labs.loc.gov/

The source data includes newspaper page images and associated JSON records containing image links and other page information.

### 2.1. Obtaining the JSON Files

The JSON files were downloaded manually from the Library of Congress API.

The following query was used to retrieve records from the Chronicling America collection:

```text
https://www.loc.gov/collections/chronicling-america/?c=160&dates=1956&fa=online-format:image&fo=json
```

The query parameters specify:

- `c=160`: Collection identifier.
- `dates=1956`: Newspaper publication year.
- `fa=online-format:image`: Filter for image-format resources.
- `fo=json`: Return results in JSON format.

The downloaded JSON files should be placed in:

```text
json_files/
```

Multiple JSON files may be required to cover the available records. All relevant downloaded JSON files are processed by the URL extraction script.

The raw JSON files are not committed to GitHub.

## 3. Repository Structure

The repository contains the source code and configuration files required to reproduce the data extraction process.

```text
project/
│
├── data_extraction_codes/
│   ├── 1_allJson_to_imageLinks.py
│   └── 2_imageLinks_to_images.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── metadata.py
│   ├── model.py
│   └── 3_process_dataset.py
│
├── tests/
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
```

The following directories and files are generated or obtained locally and are not included in the GitHub repository:

```text
project/
│
├── json_files/                  # Downloaded source JSON files
├── downloaded_newspapers/       # Downloaded newspaper page images
├── newspaper_pages_links.txt    # Extracted image URLs
├── model_weights/               # Pretrained model checkpoint
│   └── model_final.pth
├── detectron2/                  # Detectron2 source repository
├── newspaper_env/               # Python virtual environment
│
└── dataset/                     # Generated dataset
    ├── images/
    ├── metadata/
    └── completed_pages/
```

These files are either large, generated, or environment-specific. Instructions for obtaining or generating them are provided below.

## 4. Environment Setup

The pipeline was developed and tested on a MacBook with an Apple Silicon processor (M2).

### 4.1. Python Environment

Python 3.11 was used to create the virtual environment.

```bash
python3.11 -m venv newspaper_env
source newspaper_env/bin/activate
```

Verify the Python version:

```bash
python --version
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### 4.2. Install PyTorch and Torchvision

Install PyTorch and Torchvision:

```bash
python -m pip install torch torchvision
```

Verify the installation and Apple Metal Performance Shaders (MPS) availability:

```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

MPS was used to accelerate model inference on Apple Silicon hardware.

### 4.3. Install Detectron2 Dependencies

Install the required supporting packages:

```bash
python -m pip install ninja cython pycocotools fvcore iopath
```

### 4.4. Install Detectron2

Detectron2 is used for object detection. Its source code is not included in this repository and must be cloned separately.

From the project root:

```bash
git clone https://github.com/facebookresearch/detectron2.git
```

Install Detectron2 from its source directory:

```bash
cd detectron2
CC=clang CXX=clang++ python -m pip install -e . --no-build-isolation
cd ..
```

The `detectron2/` directory should be located in the project root because the project configuration references it locally.

Verify the installation:

```bash
python -c "from detectron2.config import get_cfg; print('Detectron2 installed successfully')"
```

**Note:** Detectron2 is installed as an editable local package. If the project is moved to another location, its editable installation may need to be recreated.

## 5. Model Checkpoint

The pipeline uses the pretrained Newspaper Navigator object detection model, released by the Library of Congress.

Create the model weights directory:

```bash
mkdir -p model_weights
```

Download the checkpoint directly from the official Newspaper Navigator GitHub release:

```bash
curl -L "https://github.com/LibraryOfCongress/newspaper-navigator/releases/download/v1.0.0/model_final.pth" -o model_weights/model_final.pth
```

The checkpoint should be located at:

```text
model_weights/model_final.pth
```

The checkpoint is not included in this repository because of its large file size.

The model configuration is based on:

- Architecture: Faster R-CNN
- Backbone: ResNet-50 with Feature Pyramid Network (FPN)
- Configuration: `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`
- Number of classes: 7
- Confidence threshold: 0.5
- Inference device: MPS on supported Apple Silicon hardware

The model and its configuration are loaded through `src/model.py` and `src/config.py`.

## 6. Data Acquisition

### 6.1. Extract Image URLs

The script:

```text
data_extraction_codes/1_allJson_to_imageLinks.py
```

processes the downloaded source JSON files and extracts the newspaper page image URLs.

The resulting links are consolidated into:

```text
newspaper_pages_links.txt
```

Each entry associates an image filename with its corresponding URL.

### 6.2. Download Newspaper Page Images

The script:

```text
data_extraction_codes/2_imageLinks_to_images.py
```

downloads the newspaper page images using the extracted URLs.

The final version of the downloader uses multithreading to download multiple images concurrently, improving the efficiency of the data acquisition process.

The downloaded images are stored in:

```text
downloaded_newspapers/
```

The downloader is designed to avoid downloading files that already exist locally.

The original newspaper page images are preserved at their full resolution because the object detection model identifies bounding boxes on the complete newspaper pages.

**Important:** Complete the image downloading process before starting object detection. Do not run the downloader and the processing pipeline simultaneously on the same source directory.

## 7. Object Detection and Image Extraction

The main processing script is:

```text
src/3_process_dataset.py
```

It processes the downloaded newspaper pages using the Newspaper Navigator object detection model.

For each page, the pipeline:

1. Loads the original newspaper image.
2. Runs object detection using the pretrained model.
3. Retrieves the detected object categories, confidence scores, and bounding boxes.
4. Excludes detections classified as `Headline`.
5. Crops the remaining detected objects from the original full-resolution page.
6. Saves the crops in category-specific directories.
7. Generates a JSON metadata file for the processed page.
8. Creates a completion marker to prevent unnecessary reprocessing.

### 7.1. Detection Categories

The pretrained model identifies seven categories:

| Model class | Category | Included in output |
|---|---|---|
| 0 | Illustration/Photograph | Yes |
| 1 | Photograph | Yes |
| 2 | Comics/Cartoon | Yes |
| 3 | Editorial Cartoon | Yes |
| 4 | Map | Yes |
| 5 | Headline | No |
| 6 | Advertisement | Yes |

Headline detections are excluded from the generated dataset. Therefore, six categories are saved.

The confidence threshold is set to 0.5.

### 7.2. Output Dataset Structure

The generated dataset follows a category-first directory structure:

```text
dataset/
│
├── images/
│   ├── illustration_photograph/
│   ├── photograph/
│   ├── comics_cartoon/
│   ├── editorial_cartoon/
│   ├── map/
│   └── advertisement/
│
├── metadata/
│   └── per_page/
│
└── completed_pages/
```

All crops belonging to the same category are stored together, regardless of their source newspaper page.

Each crop filename identifies its source page and category.

For example:

```text
1956-01-05_page_01_photograph_003.jpg
```

This filename indicates:
- Newspaper date: January 5, 1956
- Page number: 01
- Category: Photograph
- Detection index: 003

### 7.3. Metadata

A separate JSON metadata file is generated for each processed newspaper page.

The metadata records information associated with the extracted objects, including:

- Source newspaper filename
- Newspaper date
- Page number
- Crop filename
- Detected category
- Detection confidence score
- Bounding box coordinates

Metadata is stored in:

```text
dataset/metadata/per_page/
```

The per-page metadata structure preserves the relationship between extracted crops and their original newspaper pages.

### 7.4. Completion Markers and Resumability

The pipeline creates completion markers in:

```text
dataset/completed_pages/
```

These markers indicate that a newspaper page has been processed successfully.

When the script is run again, pages with existing completion markers are skipped. This allows processing to resume without repeating completed pages.

If a page needs to be processed again, its corresponding completion marker must be removed. Existing crops and metadata for that page should also be cleaned or regenerated to avoid duplicate outputs.

## 8. Running the Pipeline

After completing the environment setup, obtaining the model checkpoint, downloading the source images, and preparing the required directories, run the processing script from the project root.

To test the pipeline on 10 pages:

```bash
python -m src.3_process_dataset --limit 10
```

To process up to 1,000 pages:

```bash
python -m src.3_process_dataset --limit 1000
```

The `--limit` argument controls the maximum number of pages selected for processing in that run. It is not a fixed limit on the total dataset size.

The pipeline can be run again to continue processing remaining pages, as completed pages are skipped using their completion markers.

## 9. Important Reproducibility Notes

- Run commands from the project root directory.
- Activate the correct Python virtual environment before installing packages or running the pipeline.
- Ensure that the Detectron2 editable installation points to the `detectron2/` directory inside the current project.
- Ensure that `model_weights/model_final.pth` is available before starting object detection.
- Ensure that all source newspaper images have finished downloading before processing.
- The `dataset/` directory contains generated outputs and does not need to be committed to GitHub.
- If changing the output directory structure or processing logic, old completion markers and generated outputs may need to be cleared before running the updated pipeline.
- The pipeline uses the original newspaper pages for cropping rather than resizing the pages before saving detected objects.

## 10. Files Excluded from GitHub

The following files and directories are excluded from version control:

| File or directory | Reason |
|---|---|
| `newspaper_env/` | Python virtual environment |
| `detectron2/` | External source repository |
| `model_weights/` | Large pretrained model checkpoint |
| `downloaded_newspapers/` | Original downloaded newspaper images |
| `json_files/` | Downloaded source JSON files |
| `dataset/` | Generated crops, metadata, and completion markers |
| `newspaper_pages_links.txt` | Generated list of image URLs |
| `__pycache__/` | Python cache files |
| `.DS_Store` | macOS metadata |

These files are either generated during the process or can be obtained separately by following the instructions in this README.

## 11. Acknowledgements

This project builds on resources and tools made available by:

- Library of Congress – Chronicling America
- Library of Congress – Newspaper Navigator
- Facebook Research – Detectron2
- PyTorch

The Newspaper Navigator pretrained model is used for detecting and categorizing visual elements in historical newspaper pages.