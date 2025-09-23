# app.py
import io
import os
import zipfile
from typing import Any, Dict

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch

# ---- Compatibility shim for older Torch versions ----
try:
    # transformers>=4.56 may access torch.compiler.is_compiling()
    if not hasattr(torch, "compiler"):
        class _TorchCompilerShim:  # minimal API expected by transformers
            @staticmethod
            def is_compiling():
                return False
        torch.compiler = _TorchCompilerShim()  # type: ignore[attr-defined]
    elif not hasattr(torch.compiler, "is_compiling"):
        def _is_compiling_false():
            return False
        torch.compiler.is_compiling = _is_compiling_false  # type: ignore[attr-defined]
except Exception:
    pass

# ---- Image preprocess helpers ----
def resize_and_pad_to_square(img: Image.Image, target_side: int = 1024) -> Image.Image:
    """Resize image so longest side == target_side, then pad to a square (target_side x target_side)."""
    w, h = img.size
    if w == 0 or h == 0:
        return img
    scale = target_side / max(w, h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    if new_w == target_side and new_h == target_side:
        return img_resized
    # pad
    padded = Image.new("RGB", (target_side, target_side), color=(0, 0, 0))
    offset = ((target_side - new_w) // 2, (target_side - new_h) // 2)
    padded.paste(img_resized, offset)
    return padded

# ---- Helper: load HF pipeline (cached) ----
@st.cache_resource
def load_sam_pipeline(device, model_id):
    """
    Load Hugging Face mask-generation pipeline for SAM2.
    device: 0 for GPU (cuda:0), -1 for CPU
    model_id: HF model id to load
    """
    from transformers import pipeline
    # Prefer slow image processor to avoid issues in fast path on some envs
    processor = None
    try:
        from transformers import AutoImageProcessor
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        except TypeError:
            processor = AutoImageProcessor.from_pretrained(model_id)
    except Exception:
        pass

    # model name examples: facebook/sam2-hiera-large or facebook/sam2.1-hiera-large
    return pipeline(
        "mask-generation",
        model=model_id,
        device=device,
        image_processor=processor,
    )

# ---- Utility functions ----
def pil_to_np_uint8(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return arr

def masks_to_numpy_list(masks_any):
    """Converts masks (list of tensors/arrays) to list of boolean numpy arrays."""
    out = []
    for m in masks_any:
        # torch Tensor?
        if hasattr(m, "cpu") and hasattr(m, "numpy"):
            m = m.cpu().numpy()
        arr = np.asarray(m)
        # sometimes masks are floats/0..1; threshold to bool
        if arr.dtype != bool:
            arr = arr > 0.5
        out.append(arr.astype(bool))
    return out

def visualize_overlay_numpy(image_np: np.ndarray, masks_list, alpha=0.5, random_seed=None):
    """Return overlay image (uint8) by blending random colors for each mask into image_np"""
    if random_seed is not None:
        np.random.seed(random_seed)
    H, W, C = image_np.shape
    overlay = image_np.astype(np.float32) / 255.0
    for mask in masks_list:
        color = np.random.rand(3)
        # blend only on mask pixels
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    return overlay

def combined_masks_image(masks_list, random_seed=None):
    """Create a combined colorful image showing all masks (no original photo)"""
    if random_seed is not None:
        np.random.seed(random_seed)
    H, W = masks_list[0].shape
    combined = np.zeros((H, W, 3), dtype=np.float32)
    for mask in masks_list:
        color = np.random.rand(3)
        # add color contribution
        for c in range(3):
            combined[:, :, c] += mask.astype(float) * color[c]
    combined = np.clip(combined, 0, 1)
    combined_img = (combined * 255).astype(np.uint8)
    return combined_img

def make_zip_bytes(file_map: Dict[str, bytes]) -> bytes:
    """Given a dict filename -> bytes, return zip file bytes"""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        for fname, b in file_map.items():
            zf.writestr(fname, b)
    bio.seek(0)
    return bio.read()

# ---- Streamlit UI ----
st.set_page_config(layout="wide", page_title="SAM 2 Segmentation · Streamlit")

st.title("SAM 2 Segmentation (facebook/sam2-hiera-large)")
st.markdown(
    "Upload an image, adjust automatic mask generation settings, run segmentation, and download results."
)

# Sidebar: runtime & model
st.sidebar.header("Runtime and model")

# Decide device: CUDA if available, else CPU. Allow override via SAM_DEVICE env (cpu|cuda).
device_env = os.getenv("SAM_DEVICE", "").lower()
device_selected = -1
device_label = "CPU"

try:
    if device_env == "cuda" and hasattr(torch, "cuda") and torch.cuda.is_available():
        device_selected = 0
        device_label = "CUDA:0"
    elif device_env == "cpu":
        device_selected = -1
        device_label = "CPU"
    else:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            device_selected = 0
            device_label = "CUDA:0"
        else:
            device_selected = -1
            device_label = "CPU"
except Exception:
    device_selected = -1
    device_label = "CPU"

st.sidebar.info(f"Device: {device_label}")

# Model selection via env var (optional)
default_model_id = "facebook/sam2-hiera-large"
model_id = os.getenv("SAM_MODEL_ID", default_model_id)
st.sidebar.caption(f"Model: {model_id}")

# Load model (cached)
with st.spinner("Loading SAM2 pipeline (this may take a dozen seconds)..."):
    pipe = load_sam_pipeline(device=device_selected, model_id=model_id)

# Sidebar: AMG knobs UI
st.sidebar.header("Automatic mask generation settings")

# Points sampling density
points_per_side = st.sidebar.slider(
    "Points per side (grid density)",
    min_value=8,
    max_value=128,
    value=32,
    step=8,
    help="Sampling grid density. Lower is faster with fewer masks; higher finds more masks but is slower."
)
# pred IoU threshold
pred_iou_thresh = st.sidebar.slider(
    "Predicted IoU threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.88,
    step=0.01,
    help="Keep masks with predicted IoU ≥ this value. Higher yields fewer, cleaner masks."
)
# stability score
stability_score_thresh = st.sidebar.slider(
    "Stability score threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.95,
    step=0.01,
    help="Keep masks stable under small perturbations. Higher yields fewer masks."
)
# Disable multi-scale crops on CPU for stability
crops_n_layers = 0
# min mask area (pixels)
min_mask_region_area = st.sidebar.number_input(
    "Minimum mask area (px)",
    min_value=0,
    max_value=100000,
    value=100,
    help="Discard masks smaller than this pixel area."
)
# box NMS
box_nms_thresh = st.sidebar.slider(
    "Box NMS threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01,
    help="Threshold for non‑maximum suppression to reduce duplicate boxes."
)

# Output choices
st.sidebar.header("Outputs")
show_overlay = st.sidebar.checkbox("Overlay: image + masks", value=True)
show_masks_grid = st.sidebar.checkbox("Masks grid (separate)", value=False)
show_combined_masks = st.sidebar.checkbox("Combined masks (single image)", value=True)
include_overlays_in_zip = st.sidebar.checkbox("Include overlay PNGs in ZIP", value=False)

# Upload image
st.header("1) Upload image")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=False)
if uploaded is None:
    st.info("Upload an image to run segmentation.")
    st.stop()

# Read image
image = Image.open(uploaded).convert("RGB")
image_np_uint8 = pil_to_np_uint8(image)

st.subheader("Preview")
st.image(image_np_uint8, width="stretch")

# Run button
if st.button("Run segmentation"):
    PPB_LIMIT = 64 if device_selected == -1 else 128
    ppb_clamped = PPB_LIMIT

    params = {
        # we'll try to pass these names — HF pipeline expects many of them (see docs)
        "points_per_side": int(points_per_side),
        "points_per_batch": ppb_clamped,
        "pred_iou_thresh": float(pred_iou_thresh),
        "stability_score_thresh": float(stability_score_thresh),
        # HF naming sometimes uses 'crops_n_layers' (plural) — we pass that name
        "crops_n_layers": int(crops_n_layers),
        "min_mask_region_area": int(min_mask_region_area),
        "box_nms_thresh": float(box_nms_thresh)
    }

    st.info("Running mask generation… This may take from a few seconds up to a minute, depending on image size and settings.")
    with st.spinner("Generating masks..."):
        # Try the full params set; fall back if pipeline raises errors
        try:
            outputs = pipe(image, num_workers=0, batch_size=1, **params)
        except (TypeError, RuntimeError, StopIteration) as e:
            st.warning("Some settings were not supported. Retrying with safe defaults and resized input.")
            safe_image = resize_and_pad_to_square(image, 1024)
            core_params = {
                "points_per_side": 32,
                "points_per_batch": ppb_clamped,
                "pred_iou_thresh": float(pred_iou_thresh),
                "crops_n_layers": int(crops_n_layers),
            }
            outputs = pipe(safe_image, num_workers=0, batch_size=1, **core_params)

    # outputs is a dict: {'masks': [..], 'scores': [..]}
    masks_raw = outputs.get("masks", [])
    scores = outputs.get("scores", [None] * len(masks_raw))
    st.success(f"Generated {len(masks_raw)} masks.")

    # Convert to boolean numpy masks
    masks_bool = masks_to_numpy_list(masks_raw)

    # Show overlay
    if show_overlay:
        try:
            overlay_img = visualize_overlay_numpy(image_np_uint8, masks_bool, alpha=0.5, random_seed=0)
            st.subheader("Overlay")
            st.image(overlay_img, width="stretch")
        except Exception as e:
            st.error(f"Failed to make overlay: {e}")

    # Show masks grid
    if show_masks_grid:
        st.subheader("Masks grid")
        cols = st.columns(3)
        for i, mask in enumerate(masks_bool):
            col = cols[i % 3]
            # convert mask to uint8 image
            mask_img = (mask.astype(np.uint8) * 255)
            col.image(mask_img, caption=f"mask {i} (score={scores[i]:.2f})" if scores[i] is not None else f"mask {i}")

    # Show combined masks (no original image)
    if show_combined_masks:
        combined_img = combined_masks_image(masks_bool, random_seed=0)
        st.subheader("Combined masks")
        st.image(combined_img, width="stretch")

    # Prepare ZIP for download
    st.subheader("Download masks")
    files_dict = {}  # filename -> bytes
    for i, mask in enumerate(masks_bool):
        # save mask png in memory
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
        b = io.BytesIO()
        mask_img.save(b, format="PNG")
        files_dict[f"mask_{i}.png"] = b.getvalue()

    if include_overlays_in_zip and show_overlay:
        b = io.BytesIO()
        Image.fromarray(overlay_img).save(b, format="PNG")
        files_dict["overlay.png"] = b.getvalue()
    # add combined image
    if show_combined_masks:
        b = io.BytesIO()
        Image.fromarray(combined_img).save(b, format="PNG")
        files_dict["combined_masks.png"] = b.getvalue()

    zip_bytes = make_zip_bytes(files_dict)
    st.download_button("Download masks (ZIP)", data=zip_bytes, file_name="sam2_masks.zip", mime="application/zip")

    # Optionally show best mask only
    if len(scores) > 0 and any(s is not None for s in scores):
        best_idx = int(np.nanargmax(np.array([s if s is not None else -1.0 for s in scores])))
        st.subheader(f"Best mask (index {best_idx}, score={scores[best_idx]:.3f})")
        st.image((masks_bool[best_idx].astype(np.uint8) * 255), width="content")
