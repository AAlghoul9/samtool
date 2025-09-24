## Setup

```bash
# Create venv with Python 3.12
python3.12 -m venv .venv312
source .venv312/bin/activate

# Upgrade basics
python -m pip install --upgrade pip setuptools wheel

# Install project deps (no torch here)
pip install -r requirements.txt

# Install PyTorch CPU wheels
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

## Run the app

```bash
source .venv312/bin/activate
STREAMLIT_SERVER_PORT=8502 streamlit run app.py --server.headless true --server.port 8502 --server.enableXsrfProtection=false --server.enableCORS=false --browser.gatherUsageStats false
```

In the app sidebar:
- Enable "Compute DINOv2 embeddings for object crops" to export embeddings
- Choose variant (e.g., dinov2_vitb14)
- Click "Run segmentation" then "Download results (ZIP)"

## Read embeddings

From a ZIP:
```bash
source .venv312/bin/activate
python read_dinov2_embeddings.py --zip /path/to/sam2_results.zip --show-catalog
```

From a directory (unzipped or app-saved folder):
```bash
python read_dinov2_embeddings.py --npy /path/to/sam2_results-2/dinov2_embeddings.npy
```

Export arrays:
```bash
python read_dinov2_embeddings.py --zip /path/to/sam2_results.zip --export-prefix out/embeds
```

## Visualize embeddings

```bash
python visualize_dinov2_embeddings.py --dir /path/to/sam2_results-2
# Saves dinov2_similarity_heatmap.png and dinov2_pca_scatter.png into the same directory
```

## Find closest pair (cosine)

From a directory:
```bash
python closest_dinov2_pair.py --dir /path/to/sam2_results-2
```

From a zip:
```bash
python closest_dinov2_pair.py --zip /path/to/sam2_results.zip
```

From a standalone npy:
```bash
python closest_dinov2_pair.py --npy /path/to/dinov2_embeddings.npy
```

Output includes object filenames and DINOv2 variant when available.


