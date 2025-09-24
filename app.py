# app.py
import io
import os
import zipfile
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T
import zipfile

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

# ---- DINOv2 helpers ----
@st.cache_resource
def load_dinov2_model(variant: str, use_cuda: bool):
    """Load a DINOv2 backbone via torch.hub and move to desired device.
    variant examples: 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
    """
    device = torch.device("cuda:0") if use_cuda and hasattr(torch, "cuda") and torch.cuda.is_available() else torch.device("cpu")
    model = torch.hub.load('facebookresearch/dinov2', variant)
    model.eval()
    model.to(device)
    return model, device

def preprocess_for_dinov2(pil_img: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize and pad to square of given size, then convert to normalized tensor."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_sq = resize_and_pad_to_square(pil_img, size)
    tensor = T.ToTensor()(img_sq)
    tensor = T.Normalize(mean=mean, std=std)(tensor)
    return tensor

def load_embeddings_from_filelike(file_like) -> Dict[str, np.ndarray]:
    """Read embeddings from an uploaded .npy or .zip file.
    Returns dict with keys: indices (int32 [N]), features (float32 [N,D])
    """
    name = getattr(file_like, "name", "")
    data: Dict[str, np.ndarray]
    if name.endswith(".npy") or name.endswith(".npz"):
        content = file_like.read()
        arr = np.load(io.BytesIO(content), allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            arr = arr.item()
        if not isinstance(arr, dict):
            raise ValueError("NPY must contain a dict with 'indices' and 'features'")
        data = {"indices": np.asarray(arr.get("indices")), "features": np.asarray(arr.get("features"))}
        return data
    if name.endswith(".zip"):
        content = file_like.read()
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            if "dinov2_embeddings.npy" not in zf.namelist():
                raise FileNotFoundError("dinov2_embeddings.npy not found in ZIP")
            with zf.open("dinov2_embeddings.npy") as f:
                arr = np.load(io.BytesIO(f.read()), allow_pickle=True).item()
                data = {"indices": np.asarray(arr.get("indices")), "features": np.asarray(arr.get("features"))}
                return data
    raise ValueError("Unsupported file type. Upload .npy or .zip produced by the app.")

def cosine_similarity_matrix(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between rows of A and B."""
    a = features_a.astype(np.float32)
    b = features_b.astype(np.float32)
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a @ b.T

def run_sam_generate_masks(input_image: Image.Image, pipe, params: Dict[str, Any], ppb_clamped: int) -> Dict[str, Any]:
    """Run SAM2 pipeline with current params and safe fallback."""
    try:
        outputs = pipe(input_image, num_workers=0, batch_size=1, **params)
    except (TypeError, RuntimeError, StopIteration):
        safe_image = resize_and_pad_to_square(input_image, 1024)
        core_params = {
            "points_per_side": int(params.get("points_per_side", 32)),
            "points_per_batch": ppb_clamped,
            "pred_iou_thresh": float(params.get("pred_iou_thresh", 0.88)),
            "crops_n_layers": int(params.get("crops_n_layers", 0)),
        }
        outputs = pipe(safe_image, num_workers=0, batch_size=1, **core_params)
    return outputs

def embed_object_crops(input_image: Image.Image, mask_list: list, dinov2_model, dinov2_device, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices [N], features [N,D]) for object crops defined by masks."""
    feats: list = []
    idxs: list = []
    for i, mask in enumerate(mask_list):
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue
        top, left = int(ys.min()), int(xs.min())
        bottom, right = int(ys.max()) + 1, int(xs.max()) + 1
        rgb_crop = input_image.crop((left, top, right, bottom)).convert("RGB")
        tensor = preprocess_for_dinov2(rgb_crop, size=int(crop_size)).unsqueeze(0).to(dinov2_device)
        with torch.no_grad():
            f = dinov2_model(tensor)
        if isinstance(f, dict):
            f = f.get('x', next((v for v in f.values() if isinstance(v, torch.Tensor)), None))
        if isinstance(f, torch.Tensor):
            vec = f.squeeze(0).detach().cpu().float().numpy()
            feats.append(vec)
            idxs.append(i)
    if len(feats) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)
    return np.array(idxs, dtype=np.int32), np.stack(feats, axis=0).astype(np.float32)

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

def create_object_cutout_rgba(src_image: Image.Image, mask_bool: np.ndarray, tight_crop: bool = True) -> Image.Image:
    """Create an RGBA image of the object masked from src_image.
    - src_image: PIL RGB or RGBA image that matches mask shape (H, W)
    - mask_bool: numpy boolean mask (H, W)
    - tight_crop: if True, crop to the mask's bounding box
    Returns a PIL.Image in RGBA.
    """
    if mask_bool.dtype != bool:
        mask_bool = mask_bool.astype(bool)

    # Convert source to RGBA
    rgba = src_image.convert("RGBA")
    w, h = rgba.size
    # Ensure mask matches image size
    if mask_bool.shape != (h, w):
        # Best-effort fallback: return full transparent image of src size
        return Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # Alpha channel from mask
    alpha = (mask_bool.astype(np.uint8) * 255)
    r, g, b, _ = rgba.split()
    composed = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))

    if tight_crop:
        ys, xs = np.where(mask_bool)
        if ys.size == 0 or xs.size == 0:
            return composed  # empty mask; return as-is
        top, left = int(ys.min()), int(xs.min())
        bottom, right = int(ys.max()) + 1, int(xs.max()) + 1
        composed = composed.crop((left, top, right, bottom))
    return composed

def make_zip_bytes(file_map: Dict[str, bytes]) -> bytes:
    """Given a dict filename -> bytes, return zip file bytes"""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        for fname, b in file_map.items():
            zf.writestr(fname, b)
    bio.seek(0)
    return bio.read()

# ---- Streamlit UI ----
# Allow uploads from network URLs without XSRF in local/dev scenarios
try:
    st.set_option("server.enableXsrfProtection", False)
    st.set_option("server.enableCORS", False)
except Exception:
    pass

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
include_object_crops = st.sidebar.checkbox("Include object images (transparent BG) in ZIP", value=True)
object_tight_crop = st.sidebar.checkbox("Tight crop object images to bounding box", value=True)
include_masks_in_zip = st.sidebar.checkbox("Include mask PNGs in ZIP", value=False)

# Embeddings (DINOv2)
st.sidebar.header("Embeddings")
# Session state for the crops embedding toggle
if "embed_with_dinov2" not in st.session_state:
    st.session_state.embed_with_dinov2 = False

# Target first so we can disable crop-embedding toggle when needed
embedding_target = st.sidebar.selectbox(
    "Embedding target",
    options=["Objects (from masks)", "Whole image (no SAM)"],
    index=0,
    help="Choose to embed each object (requires masks) or the entire image without segmentation."
)

# If whole-image is selected, force-disable crop embeddings
disable_embed_crops = (embedding_target == "Whole image (no SAM)")
if disable_embed_crops and st.session_state.embed_with_dinov2:
    st.session_state.embed_with_dinov2 = False

# Toggle for object-crop embeddings, disabled in whole-image mode
st.sidebar.checkbox(
    "Compute DINOv2 embeddings for object crops",
    key="embed_with_dinov2",
    disabled=disable_embed_crops,
    help=("Disabled in whole-image mode." if disable_embed_crops else "Compute embeddings for each detected object crop.")
)
embed_with_dinov2 = bool(st.session_state.embed_with_dinov2)

dinov2_variant = st.sidebar.selectbox(
    "DINOv2 variant",
    options=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    index=1,
    help="Backbone size. Larger models yield stronger embeddings but are slower."
)
dinov2_crop_size = st.sidebar.slider(
    "Embedding crop size (px)",
    min_value=128,
    max_value=392,
    value=224,
    step=32,
    help="Resize+letterbox to this square size before DINOv2. Applies to objects and whole image."
)

# Upload image
st.header("1) Upload image")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=False)
if uploaded is None:
    st.info("Upload an image to run segmentation.")
    # Allow comparisons even without image if files are provided below

# Read image (only if provided)
image = None
image_np_uint8 = None
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    image_np_uint8 = pil_to_np_uint8(image)
    
    st.subheader("Preview")
    st.image(image_np_uint8, width="stretch")

# Run button
if st.button("Run segmentation"):
    # Whole-image embedding path (skip SAM) if selected
    if embed_with_dinov2 and embedding_target == "Whole image (no SAM)":
        if uploaded is None or image is None:
            st.error("Please upload an image to compute the whole-image embedding.")
            st.stop()
        st.info("Computing DINOv2 embedding for the whole image (no SAM masks)...")
        try:
            with st.spinner("Loading DINOv2 model…"):
                dinov2_model, dinov2_device = load_dinov2_model(dinov2_variant, use_cuda=(device_selected == 0))
            with st.spinner("Embedding whole image…"):
                tensor = preprocess_for_dinov2(image, size=int(dinov2_crop_size)).unsqueeze(0).to(dinov2_device)
                with torch.no_grad():
                    feats = dinov2_model(tensor)
                if isinstance(feats, dict):
                    feats = feats.get('x', next((v for v in feats.values() if isinstance(v, torch.Tensor)), None))
                if not isinstance(feats, torch.Tensor):
                    raise RuntimeError("Unexpected DINOv2 output type")
                vec = feats.squeeze(0).detach().cpu().float().numpy()

            files_dict = {}
            # Save original image
            bimg = io.BytesIO()
            image.save(bimg, format="PNG")
            files_dict["image.png"] = bimg.getvalue()
            # Save per-item vector
            bnp = io.BytesIO()
            np.save(bnp, vec)
            files_dict["whole_image_dinov2.npy"] = bnp.getvalue()
            # Save catalog and stacked file (single row)
            header = ["index", "variant", "crop_h", "crop_w", "feat_dim"]
            lines = ["\t".join(header)]
            lines.append("\t".join([
                "0",
                dinov2_variant,
                str(image.size[1]),
                str(image.size[0]),
                str(int(vec.shape[0]))
            ]))
            files_dict["dinov2_catalog.tsv"] = ("\n".join(lines)).encode("utf-8")

            stacked = {"indices": np.array([0], dtype=np.int32), "features": np.expand_dims(vec, axis=0)}
            bstack = io.BytesIO()
            np.save(bstack, stacked)
            files_dict["dinov2_embeddings.npy"] = bstack.getvalue()

            zip_bytes = make_zip_bytes(files_dict)
            st.download_button(
                "Download whole-image embedding (ZIP)",
                data=zip_bytes,
                file_name="dinov2_whole_image.zip",
                mime="application/zip"
            )
            st.success(f"Whole-image embedding computed. Feature dim: {vec.shape[0]}")
        except Exception as e:
            st.error(f"Failed to compute whole-image embedding: {e}")
        st.stop()

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

    if uploaded is None:
        st.error("Please upload an image for SAM segmentation or use the comparison tools below.")
        st.stop()

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
    st.subheader("Download masks and objects")
    files_dict = {}  # filename -> bytes
    dinov2_embeddings = []  # list of (idx, vector np.ndarray)
    dinov2_meta = []  # list of dicts with metadata
    dinov2_model = None
    dinov2_device = None
    if embed_with_dinov2 and len(masks_bool) > 0:
        with st.spinner("Loading DINOv2 model for embeddings…"):
            dinov2_model, dinov2_device = load_dinov2_model(dinov2_variant, use_cuda=(device_selected == 0))
    for i, mask in enumerate(masks_bool):
        # optionally save mask png
        if include_masks_in_zip:
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
            b = io.BytesIO()
            mask_img.save(b, format="PNG")
            files_dict[f"mask_{i}.png"] = b.getvalue()

        # optionally save object images (transparent background)
        if include_object_crops:
            try:
                obj_img = create_object_cutout_rgba(image, mask, tight_crop=object_tight_crop)
                b2 = io.BytesIO()
                obj_img.save(b2, format="PNG")
                files_dict[f"object_{i}.png"] = b2.getvalue()
            except Exception as e:
                # Skip this object if something goes wrong; continue others
                pass

        # optionally compute DINOv2 embedding on tight bounding crop from original RGB
        if embed_with_dinov2 and dinov2_model is not None:
            try:
                ys, xs = np.where(mask)
                if ys.size == 0 or xs.size == 0:
                    continue
                top, left = int(ys.min()), int(xs.min())
                bottom, right = int(ys.max()) + 1, int(xs.max()) + 1
                rgb_crop = image.crop((left, top, right, bottom)).convert("RGB")
                tensor = preprocess_for_dinov2(rgb_crop, size=int(dinov2_crop_size)).unsqueeze(0).to(dinov2_device)
                with torch.no_grad():
                    feats = dinov2_model(tensor)
                # Some hubs may return dict; handle both
                if isinstance(feats, dict):
                    if 'x' in feats:
                        feats = feats['x']
                    else:
                        # take first tensor value
                        feats = next((v for v in feats.values() if isinstance(v, torch.Tensor)), None)
                if isinstance(feats, torch.Tensor):
                    vec = feats.squeeze(0).detach().cpu().float().numpy()
                    dinov2_embeddings.append((i, vec))
                    dinov2_meta.append({
                        "index": i,
                        "variant": dinov2_variant,
                        "crop_h": rgb_crop.size[1],
                        "crop_w": rgb_crop.size[0],
                        "feat_dim": int(vec.shape[0])
                    })
                    # save per-object npy
                    bnp = io.BytesIO()
                    np.save(bnp, vec)
                    files_dict[f"object_{i}_dinov2.npy"] = bnp.getvalue()
            except Exception as e:
                # continue on errors per object
                pass

    if include_overlays_in_zip and show_overlay:
        b = io.BytesIO()
        Image.fromarray(overlay_img).save(b, format="PNG")
        files_dict["overlay.png"] = b.getvalue()
    # add combined image
    if show_combined_masks:
        b = io.BytesIO()
        Image.fromarray(combined_img).save(b, format="PNG")
        files_dict["combined_masks.png"] = b.getvalue()

    # If we computed any embeddings, save also a catalog TSV and a stacked matrix
    if embed_with_dinov2 and len(dinov2_embeddings) > 0:
        try:
            # TSV
            header = ["index", "variant", "crop_h", "crop_w", "feat_dim"]
            lines = ["\t".join(header)]
            for row in dinov2_meta:
                lines.append("\t".join(str(row[k]) for k in header))
            files_dict["dinov2_catalog.tsv"] = ("\n".join(lines)).encode("utf-8")
            # Stacked features
            indices = [idx for idx, _ in dinov2_embeddings]
            mat = np.stack([vec for _, vec in dinov2_embeddings], axis=0)
            bnp = io.BytesIO()
            np.save(bnp, {"indices": np.array(indices, dtype=np.int32), "features": mat})
            files_dict["dinov2_embeddings.npy"] = bnp.getvalue()
        except Exception:
            pass

    zip_bytes = make_zip_bytes(files_dict)
    st.download_button("Download results (ZIP)", data=zip_bytes, file_name="sam2_results.zip", mime="application/zip")

    # Optionally show best mask only
    if len(scores) > 0 and any(s is not None for s in scores):
        best_idx = int(np.nanargmax(np.array([s if s is not None else -1.0 for s in scores])))
        st.subheader(f"Best mask (index {best_idx}, score={scores[best_idx]:.3f})")
        st.image((masks_bool[best_idx].astype(np.uint8) * 255), width="content")

# ---- Embedding comparisons (below segmentation) ----
st.header("2) Embedding comparisons")
st.caption("Upload two images; we'll compute DINOv2 whole-image and object embeddings for both, then compare.")

imgA = st.file_uploader("Image A", type=["jpg", "jpeg", "png", "bmp"], key="pair_imgA")
imgB = st.file_uploader("Image B", type=["jpg", "jpeg", "png", "bmp"], key="pair_imgB")
topk_pairs = st.number_input("Top-K object matches to show", min_value=1, max_value=100, value=10, step=1)

col_t1, col_t2, col_t3, col_t4 = st.columns(4)
with col_t1:
    whole_dup_thresh = st.number_input("Whole dup threshold", min_value=0.0, max_value=1.0, value=0.95, step=0.01, help="cosine similarity ≥ this → likely duplicate/copy")
with col_t2:
    whole_sim_thresh = st.number_input("Whole similar threshold", min_value=0.0, max_value=1.0, value=0.85, step=0.01, help="cosine similarity ≥ this → similar")
with col_t3:
    obj_sim_thresh = st.number_input("Object match sim ≥", min_value=0.0, max_value=1.0, value=0.80, step=0.01, help="object pair considered a match if cos sim ≥ this")
with col_t4:
    obj_cover_thresh = st.number_input("Object coverage ≥", min_value=0.0, max_value=1.0, value=0.50, step=0.05, help="fraction of A's objects with a match in B to claim similarity")

if imgA is not None and imgB is not None and st.button("Compare two images (whole + objects)"):
    try:
        with st.spinner("Loading DINOv2 and SAM2…"):
            dinov2_model, dinov2_device = load_dinov2_model(dinov2_variant, use_cuda=(device_selected == 0))
        imA = Image.open(imgA).convert("RGB")
        imB = Image.open(imgB).convert("RGB")
        # Whole-image embeddings
        tA = preprocess_for_dinov2(imA, size=int(dinov2_crop_size)).unsqueeze(0).to(dinov2_device)
        tB = preprocess_for_dinov2(imB, size=int(dinov2_crop_size)).unsqueeze(0).to(dinov2_device)
        with torch.no_grad():
            fA = dinov2_model(tA)
            fB = dinov2_model(tB)
        if isinstance(fA, dict):
            fA = fA.get('x', next((v for v in fA.values() if isinstance(v, torch.Tensor)), None))
        if isinstance(fB, dict):
            fB = fB.get('x', next((v for v in fB.values() if isinstance(v, torch.Tensor)), None))
        vA = fA.squeeze(0).detach().cpu().float().numpy()
        vB = fB.squeeze(0).detach().cpu().float().numpy()
        whole_sim = float((vA @ vB) / ((np.linalg.norm(vA) + 1e-8) * (np.linalg.norm(vB) + 1e-8)))
        st.subheader("Whole-image similarity")
        whole_dist = 1.0 - whole_sim
        st.write(f"Cosine similarity: {whole_sim:.6f}   Cosine distance: {whole_dist:.6f}")
        if whole_sim >= whole_dup_thresh:
            st.success("Judgment: Likely duplicate/copy (≥ dup threshold)")
        elif whole_sim >= whole_sim_thresh:
            st.warning("Judgment: Similar (≥ similar threshold)")
        else:
            st.info("Judgment: Dissimilar (< similar threshold)")

        # Objects via SAM2
        PPB_LIMIT = 64 if device_selected == -1 else 128
        ppb_clamped = PPB_LIMIT
        params = {
            "points_per_side": int(points_per_side),
            "points_per_batch": ppb_clamped,
            "pred_iou_thresh": float(pred_iou_thresh),
            "stability_score_thresh": float(stability_score_thresh),
            "crops_n_layers": int(crops_n_layers),
            "min_mask_region_area": int(min_mask_region_area),
            "box_nms_thresh": float(box_nms_thresh)
        }
        with st.spinner("Running SAM2 on Image A…"):
            outA = run_sam_generate_masks(imA, pipe, params, ppb_clamped)
        with st.spinner("Running SAM2 on Image B…"):
            outB = run_sam_generate_masks(imB, pipe, params, ppb_clamped)
        masksA = masks_to_numpy_list(outA.get("masks", []))
        masksB = masks_to_numpy_list(outB.get("masks", []))
        with st.spinner("Embedding object crops…"):
            idxA, featsA = embed_object_crops(imA, masksA, dinov2_model, dinov2_device, int(dinov2_crop_size))
            idxB, featsB = embed_object_crops(imB, masksB, dinov2_model, dinov2_device, int(dinov2_crop_size))
        if featsA.size == 0 or featsB.size == 0:
            st.warning("One of the images has no object embeddings; cannot compute object-to-object similarities.")
        else:
            simAB = cosine_similarity_matrix(featsA, featsB)
            # Flatten and get top-K pairs
            flat = simAB.ravel()
            k = min(int(topk_pairs), flat.size)
            top_idx = np.argpartition(-flat, kth=k-1)[:k]
            # convert flat indices to (i,j)
            rows = []
            num_cols = simAB.shape[1]
            for rank, fi in enumerate(top_idx, start=1):
                i = int(fi // num_cols)
                j = int(fi % num_cols)
                sim = float(simAB[i, j])
                rows.append({
                    "rank": rank,
                    "A_index": int(idxA[i]),
                    "B_index": int(idxB[j]),
                    "cos_sim": sim,
                    "cos_dist": 1.0 - sim,
                })
            # Sort rows by similarity desc and show top-K in order
            rows.sort(key=lambda r: r["cos_sim"], reverse=True)
            st.subheader("Top object similarities (A vs B)")
            st.dataframe(rows, use_container_width=True)

            # Coverage metric: fraction of A's objects that have at least one B match ≥ obj_sim_thresh
            best_per_A = simAB.max(axis=1)
            coverage = float((best_per_A >= obj_sim_thresh).mean()) if best_per_A.size else 0.0
            st.write(f"Object coverage (A→B, sim≥{obj_sim_thresh:.2f}): {coverage*100:.1f}%")
            if coverage >= obj_cover_thresh:
                st.success("Object judgment: Similar (coverage ≥ threshold)")
            else:
                st.info("Object judgment: Not similar enough (coverage < threshold)")
    except Exception as e:
        st.error(f"Image-to-image comparison failed: {e}")
