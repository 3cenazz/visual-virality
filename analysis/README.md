### Codes in this folder run the analysis pipeline (embeddings, surprisal, circulation). Steps as follows:
## 1. Download requirements
Ensure your working directory is on the main `visual-virality` folder. Run the following code in your terminal to download the necessary packages. 
```bash
python -m pip install -r analysis/requirements.txt
```

## 2. Run full pipeline 
Run the `run_analysis_pipeline.py` file to construct the running distribution, calculate surprisal and circulation for the embedded images.
```bash
python -m analysis.run_analysis_pipeline
```

## Stored files
- `dataset/analysis/running_dist.parquet`: metadata plus rolling novelty scores