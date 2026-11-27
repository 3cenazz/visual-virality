### Codes in this folder run the analysis pipeline (embeddings, surprisal, circulation). Steps as follows:
## 1. Download requirements
Ensure your working directory is on the main `visual-virality` folder. Run the following code in your terminal to download the necessary packages. 
```bash
python -m pip install -r analysis/requirements.txt
```

## 1b. Point the code at the shared dataset
The embeddings are no longer in this repo. They live in the shared OneDrive dataset folder under `embeddings/`
(`embeddings_resnet18.npy`, `embeddings_resnet18_meta.csv`, `metadata_all.json`). Tell the code where your synced copy is:
```bash
export VV_DATASET_DIR="$HOME/Library/CloudStorage/OneDrive-epfl.ch/Dateien von Ecenaz Egri - dataset"
```
Adjust the folder name to how it appears on your machine. Without this, the code looks for `../dataset` next to the repo.

## 2. Run full pipeline 
Run the `run_analysis_pipeline.py` file to construct the running distribution, calculate surprisal and circulation for the embedded images.
```bash
python -m analysis.run_analysis_pipeline
```

## Stored files
- `running_dist.csv`: image metadata and computed dissimilairty and surprisal scores