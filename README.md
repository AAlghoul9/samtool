# SAM2 Segmentation — Streamlit App

Interactive demo for automatic image segmentation using the Hugging Face transformers mask-generation pipeline with the model `facebook/sam2-hiera-large`.

The app lets you:
- Upload an image
- Tweak SAM2 Automatic Mask Generation (AMG) knobs
- Run segmentation on CPU, Apple Silicon (MPS), or CUDA GPU
- Preview overlays and export masks as a ZIP

---

## Quickstart

```bash
# 1) Create and activate a virtual environment (Python 3.12 REQUIRED)
python3.12 --version   # should print Python 3.12.x
python3.12 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 2b) Install PyTorch CPU wheels (required)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 3) Run the Streamlit app
streamlit run app.py
```

Then open the URL that Streamlit prints (usually `http://localhost:8501`).

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On Streamlit Community Cloud, create a new app pointing to `app.py`.
3. No special hardware settings are required. Streamlit Cloud does not provide GPUs; the app will run on CPU and auto-detect devices if available.
4. Ensure dependencies are pinned in `requirements.txt` (already provided). First run will download the model to the ephemeral cache.

Optional configuration via environment variables:
- `SAM_MODEL_ID` (default `facebook/sam2-hiera-large`): alternate model id
- `SAM_DEVICE` (`cpu` or `cuda`): force device selection; by default the app auto-detects

If you hit memory/time limits, reduce `points_per_side` and keep defaults for `crops_n_layers`.

---

## Requirements
- Python 3.12 (required). The pinned PyTorch/torchvision wheels do not support Python 3.13.
- Internet connection on first run to download the model weights and processor from Hugging Face.
- Disk space for model cache (several GB). Downloaded to `~/.cache/huggingface` by default.

### Hardware acceleration
- Apple Silicon (MPS): Supported if your macOS/PyTorch supports MPS. The app will auto-detect and use it when “Prefer hardware acceleration” is checked.
- NVIDIA CUDA: Supported if you install a CUDA-enabled PyTorch build (see below). The default `requirements.txt` pins CPU wheels.
- CPU-only: Works everywhere; slower.

---

## Python version and setup

- macOS (Homebrew):
  ```bash
  brew install python@3.12
  python3.12 --version   # verify 3.12.x
  python3.12 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- macOS/Linux (pyenv):
  ```bash
  # install pyenv per docs, then
  pyenv install 3.12.7
  pyenv local 3.12.7
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Windows (PowerShell):
  ```powershell
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  ```

Why 3.12? If you use Python 3.13 you will see errors like:

```
ERROR: No matching distribution found for torch==2.2.2
```

Switch to Python 3.12 to use the pinned wheels.

---

## CUDA users (optional)
If you have an NVIDIA GPU and want CUDA acceleration, install PyTorch with CUDA per the official instructions instead of the CPU wheels in `requirements.txt`.

Example for CUDA 12.1 wheels:
```bash
# Activate your venv first
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Then install the rest of the deps (skip torch/torchvision duplicates)
pip install -r requirements.txt --no-deps
pip install "transformers==4.56.1" "streamlit==1.49.1" "Pillow>=10.0.0" "numpy<2.0" "matplotlib>=3.8"
```
Adjust CUDA version and commands based on your environment: see [PyTorch](https://pytorch.org).

---

## Usage
1. Start the app: `streamlit run app.py`.
2. Upload a `.jpg`, `.jpeg`, `.png`, or `.bmp` image.
3. The app auto-detects available acceleration (CUDA/MPS) and falls back to CPU.
4. Tweak AMG knobs (e.g., `points_per_side`, `pred_iou_thresh`, `crops_n_layers`).
5. Choose Embedding options in the sidebar (optional):
   - Embedding target: “Objects (from masks)” or “Whole image (no SAM)”
   - DINOv2 variant: `dinov2_vits14`, `vitb14`, `vitl14`, `vitg14`
   - Embedding crop size (applies to objects and whole image)
6. Click “Run segmentation”.
7. Inspect:
   - Overlay (image + colorful masks)
   - Masks-only grid (optional)
   - Combined masks image
8. Download a ZIP containing:
   - Object PNGs (optional)
   - `dinov2_embeddings.npy` (indices, features) and `dinov2_catalog.tsv` (if embeddings enabled)

### 2) Embedding comparisons (two images)
- Upload Image A and Image B in section “Embedding comparisons”.
- Click “Compare two images (whole + objects)”. The app will:
  - Compute whole-image DINOv2 embeddings for A and B and show cosine similarity/distance
  - Run SAM2 on both images, embed all objects with DINOv2, compute A↔B cosine similarities, and list Top‑K object matches
- Adjustable thresholds in UI:
  - Whole dup threshold: cosine ≥ threshold → “Likely duplicate/copy”
  - Whole similar threshold: cosine ≥ threshold → “Similar”, else “Dissimilar”
  - Object match sim ≥: per‑object cosine to be counted as a match
  - Object coverage ≥: fraction of A’s objects matched in B to consider objects “Similar”

---

## Notes
- First run downloads the model/processor from the Hugging Face Hub and may take a while.
- Cache location can be changed via env var `HF_HOME` if desired.
- Large images may be preprocessed (resize/pad) automatically if the pipeline rejects parameters due to memory/shape constraints.

---

## Troubleshooting
- Slow or no GPU acceleration:
  - Apple Silicon: Ensure macOS supports MPS and you’re on a recent PyTorch. The app will detect and use MPS automatically.
  - NVIDIA: Ensure a CUDA-enabled PyTorch is installed (see CUDA users section) and correct drivers are present.
- Out-of-memory or shape errors:
  - Reduce `points_per_side` / `points_per_batch`.
  - Try enabling the built-in preprocessing by leaving defaults and re-running.
- Model download issues:
  - Check internet connectivity and available disk space.
  - Optional: set `HF_HOME` to a drive with more space, e.g., `export HF_HOME=/path/with/space`.
- Import/version conflicts:
  - Use a fresh virtual environment with Python 3.12 and reinstall.

---

## Project layout
```
.
├── app.py            # Streamlit app
├── requirements.txt  # Pinned dependencies
├── COMMANDS.md       # Helpful setup & CLI usage
├── read_dinov2_embeddings.py      # Read dinov2_embeddings from ZIP/NPY
├── visualize_dinov2_embeddings.py # Make heatmap + PCA plots
├── closest_dinov2_pair.py         # Find closest pairs/top-K by cosine
└── README.md         # This file
```

---

## License
This repository is for demonstration purposes. Review the licenses of third-party models and libraries (PyTorch, transformers, Streamlit, SAM2 checkpoints) before commercial use.
