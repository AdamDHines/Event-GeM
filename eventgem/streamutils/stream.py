import h5py
import cv2
import re
import numpy as np
from typing import Tuple, List, Optional, Sequence, Callable, Union
from collections import OrderedDict
import torch
import math
from pathlib import Path
import yaml
from torch.nn import functional as F
import time
import json
import os
from skimage.metrics import structural_similarity as ssim

# ---------------------------
# HDF5 helpers
# ---------------------------
EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")

def sec_to_raw(t_sec: float, time_scale: float) -> int:
    return int(round(float(t_sec) / float(time_scale)))


def raw_to_sec(t_raw: int, time_scale: float) -> float:
    return float(t_raw) * float(time_scale)

def _find_first_dataset(g: h5py.Group, candidates: tuple[str, ...], logical_name: str):
    for key in candidates:
        if key in g and isinstance(g[key], h5py.Dataset):
            return g[key]
    for _, obj in g.items():
        if isinstance(obj, h5py.Group):
            try:
                return _find_first_dataset(obj, candidates, logical_name)
            except RuntimeError:
                pass
    raise RuntimeError(
        f"Could not find {logical_name} dataset under group '{g.name}'. "
        f"Tried names: {', '.join(candidates)}"
    )


def find_event_datasets(f: h5py.File):
    g = f["events"] if ("events" in f and isinstance(f["events"], h5py.Group)) else f
    x_ds = _find_first_dataset(g, EVENT_X_KEYS, "x")
    y_ds = _find_first_dataset(g, EVENT_Y_KEYS, "y")
    t_ds = _find_first_dataset(g, EVENT_T_KEYS, "t")
    p_ds = _find_first_dataset(g, EVENT_P_KEYS, "p")
    return x_ds, y_ds, t_ds, p_ds


def infer_resolution(x_dset: h5py.Dataset, y_dset: h5py.Dataset, chunk_size: int, full_scan: bool) -> Tuple[int, int]:
    N = x_dset.size
    if N == 0:
        return 0, 0

    max_x = 0
    max_y = 0

    if full_scan:
        for i in range(0, N, chunk_size):
            j = min(N, i + chunk_size)
            x = x_dset[i:j]
            y = y_dset[i:j]
            if x.size:
                max_x = max(max_x, int(np.max(x)))
            if y.size:
                max_y = max(max_y, int(np.max(y)))
    else:
        j = min(N, chunk_size)
        x = x_dset[0:j]
        y = y_dset[0:j]
        if x.size:
            max_x = int(np.max(x))
        if y.size:
            max_y = int(np.max(y))

    return max_y + 1, max_x + 1

# ---------------------------
# Streaming windows (Optimized)
# ---------------------------
def stream_event_windows_raw(
    hdf5_path: Path,
    dt_ms: float,
    chunk_size: int,
    time_scale: float,
    start_time_sec: Optional[float],
    max_frames: Optional[int],
):
    f = h5py.File(hdf5_path, "r")
    x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)
    N = len(t_dset)
    if N == 0:
        f.close()
        return

    dt_raw = int(round((dt_ms / 1000.0) / float(time_scale)))
    if dt_raw <= 0:
        raise ValueError(f"dt_ms too small for time_scale (dt_raw={dt_raw})")

    t0_raw = int(t_dset[0])
    tN_raw = int(t_dset[N - 1])

    if start_time_sec is None:
        w_start_raw = t0_raw
    else:
        w_start_raw = max(sec_to_raw(start_time_sec, time_scale), t0_raw)

    if w_start_raw >= tN_raw:
        print(f"[DBG] START BEYOND END: start_raw={w_start_raw}, file_end_raw={tN_raw}")
        f.close()
        return

    x_buf = np.empty(0, dtype=np.int64)
    y_buf = np.empty(0, dtype=np.int64)
    t_buf = np.empty(0, dtype=np.int64)
    p_buf = np.empty(0, dtype=np.int8)

    read_idx = 0
    frame_idx = 0
    t_buf_max = -1

    while w_start_raw < tN_raw and (max_frames is None or frame_idx < max_frames):
        w_end_raw = w_start_raw + dt_raw
        t_read0 = time.perf_counter()

        accum_x, accum_y, accum_t, accum_p = [], [], [], []
        
        if t_buf.size > 0:
            accum_x.append(x_buf)
            accum_y.append(y_buf)
            accum_t.append(t_buf)
            accum_p.append(p_buf)
            t_buf_max = int(t_buf[-1])

        while read_idx < N and (t_buf_max < w_end_raw):
            end_idx = min(N, read_idx + chunk_size)
            t_chunk = t_dset[read_idx:end_idx].astype(np.int64, copy=False)
            
            if t_chunk.size > 0:
                x_chunk = x_dset[read_idx:end_idx].astype(np.int64, copy=False)
                y_chunk = y_dset[read_idx:end_idx].astype(np.int64, copy=False)
                p_chunk = p_dset[read_idx:end_idx].astype(np.int8, copy=False)
                
                accum_x.append(x_chunk)
                accum_y.append(y_chunk)
                accum_t.append(t_chunk)
                accum_p.append(p_chunk)
                t_buf_max = int(t_chunk[-1])
            
            read_idx = end_idx

        if accum_t:
            x_buf = np.concatenate(accum_x)
            y_buf = np.concatenate(accum_y)
            t_buf = np.concatenate(accum_t)
            p_buf = np.concatenate(accum_p)
        else:
            x_buf = np.empty(0, dtype=np.int64)
            y_buf = np.empty(0, dtype=np.int64)
            t_buf = np.empty(0, dtype=np.int64)
            p_buf = np.empty(0, dtype=np.int8)

        if t_buf.size:
            in_win = (t_buf >= w_start_raw) & (t_buf < w_end_raw)
            x_win = x_buf[in_win]
            y_win = y_buf[in_win]
            t_win_raw = t_buf[in_win]
            p_win = p_buf[in_win]

            leftover = t_buf >= w_end_raw
            x_buf, y_buf, t_buf, p_buf = x_buf[leftover], y_buf[leftover], t_buf[leftover], p_buf[leftover]
            t_buf_max = int(t_buf[-1]) if t_buf.size > 0 else -1
        else:
            x_win = np.empty(0, dtype=np.int64)
            y_win = np.empty(0, dtype=np.int64)
            t_win_raw = np.empty(0, dtype=np.int64)
            p_win = np.empty(0, dtype=np.int8)

        t_read1 = time.perf_counter()
        t_read_ms = (t_read1 - t_read0) * 1000.0

        yield (
            raw_to_sec(w_start_raw, time_scale),
            raw_to_sec(w_end_raw, time_scale),
            w_end_raw,
            x_win, y_win, t_win_raw, p_win,
            frame_idx,
            t_read_ms,
        )

        w_start_raw = w_end_raw
        frame_idx += 1

        if read_idx >= N and t_buf.size == 0 and t_buf_max < w_start_raw:
            break

    f.close()

import numpy as np
import torch
from typing import Optional

@torch.no_grad()
def tencode(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    white_frame: bool = False,
    normalize: bool = True,
    device: Optional[torch.device] = None,
    return_numpy: bool = False,
):
    """
    GPU-accelerated Tencode (last-event-wins per pixel), modeled after gpu_polarity_frame_2ch.

    If device is None -> CPU numpy path (original behavior).
    If device is cuda -> builds on GPU and returns torch.Tensor by default.
      - set return_numpy=True to get a numpy array back (slower, defeats most of the point).
    Output shape: (3, H, W), float32, normalized to [0,1] if normalize else [0,255].
    """
    assert x.ndim == y.ndim == t.ndim == p.ndim == 1
    H, W = height, width
    base_val = 255.0 if white_frame else 0.0

    # --------------------
    # CPU fallback (original semantics) if no device requested
    # --------------------
    if device is None or (isinstance(device, torch.device) and device.type == "cpu"):
        n = x.shape[0]
        if n == 0:
            frame = np.full((3, H, W), base_val, dtype=np.float32)
            return frame / 255.0 if normalize else frame

        x64 = x.astype(np.int64, copy=False)
        y64 = y.astype(np.int64, copy=False)
        t64 = t.astype(np.float64, copy=False)
        p64 = p.astype(np.float64, copy=False)

        order = np.argsort(t64)
        x64, y64, t64, p64 = x64[order], y64[order], t64[order], p64[order]

        pol = (p64 > 0).astype(np.float32)
        if t64[-1] != t64[0]:
            t_norm = (t64 - t64[0]) / (t64[-1] - t64[0])
        else:
            t_norm = np.zeros_like(t64, dtype=np.float32)

        out = np.full((3, H, W), base_val, dtype=np.float32)
        valid = (x64 >= 0) & (x64 < W) & (y64 >= 0) & (y64 < H)
        if np.any(valid):
            xv = x64[valid]; yv = y64[valid]
            polv = pol[valid]; tn = t_norm[valid]
            out[0, yv, xv] = 255.0 * polv
            out[1, yv, xv] = 255.0 * (1.0 - tn)
            out[2, yv, xv] = 255.0 * (1.0 - polv)

        return (out / 255.0) if normalize else out

    # --------------------
    # GPU path
    # --------------------
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError(
            "Your PyTorch doesn't have Tensor.scatter_reduce_. "
            "Upgrade PyTorch (recommended) to use the GPU tencode path."
        )

    if x.size == 0:
        out = torch.full((3, H, W), base_val, device=device, dtype=torch.float32)
        if normalize:
            out = out / 255.0
        return out.cpu().numpy() if return_numpy else out

    # Move window to GPU
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.int64)
    y_t = torch.from_numpy(y).to(device=device, dtype=torch.int64)

    # t should be int64 raw timestamps if coming from your stream
    t_t = torch.from_numpy(t).to(device=device)
    if t_t.dtype != torch.int64:
        # If you pass float timestamps, we can still do it, but int64 raw is best for correctness.
        t_t = t_t.to(torch.int64)

    # Polarity fix (same idea as your gpu_polarity_frame_2ch)
    p_t = torch.from_numpy(p).to(device=device)
    if p_t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        p_cmp = p_t.to(torch.int16)  # avoids uint8 255 -> int8 -1 style wrap issues
    else:
        p_cmp = p_t

    valid = (x_t >= 0) & (x_t < W) & (y_t >= 0) & (y_t < H)
    if not torch.any(valid):
        out = torch.full((3, H, W), base_val, device=device, dtype=torch.float32)
        if normalize:
            out = out / 255.0
        return out.cpu().numpy() if return_numpy else out

    x_t = x_t[valid]
    y_t = y_t[valid]
    t_t = t_t[valid]
    p_cmp = p_cmp[valid]

    n = int(x_t.numel())
    flat = H * W
    lin = y_t * W + x_t  # (n,)

    # polarity in {0,1}
    pol = (p_cmp > 0).to(torch.float32)

    # Window min/max timestamp (for t_norm)
    t_min = t_t.min().to(torch.float32)
    t_max = t_t.max().to(torch.float32)
    denom = (t_max - t_min)

    # 1) per-pixel max timestamp
    min_i64 = torch.iinfo(torch.int64).min
    t_last = torch.full((flat,), min_i64, device=device, dtype=torch.int64)
    t_last.scatter_reduce_(0, lin, t_t, reduce="amax", include_self=True)

    # 2) among events that hit that max timestamp per pixel, take max event index (tie-break = "last")
    ev_idx = torch.arange(n, device=device, dtype=torch.int64)
    is_max_t = (t_t == t_last[lin])
    cand_idx = torch.where(is_max_t, ev_idx, torch.full_like(ev_idx, -1))
    last_idx = torch.full((flat,), -1, device=device, dtype=torch.int64)
    last_idx.scatter_reduce_(0, lin, cand_idx, reduce="amax", include_self=True)

    has = last_idx >= 0
    if not torch.any(has):
        out = torch.full((3, H, W), base_val, device=device, dtype=torch.float32)
        if normalize:
            out = out / 255.0
        return out

    # Build per-pixel pol_last via a unique selection mask (no duplicate writes)
    sel = is_max_t & (ev_idx == last_idx[lin])
    pol_last = torch.zeros((flat,), device=device, dtype=torch.float32)
    pol_last[lin[sel]] = pol[sel]

    # t_norm of last event per pixel (only where has events)
    if float(denom.item()) > 0.0:
        t_last_f = t_last[has].to(torch.float32)
        t_norm_last = (t_last_f - t_min) / denom
    else:
        t_norm_last = torch.zeros((int(has.sum().item()),), device=device, dtype=torch.float32)

    # Write output (3, H, W)
    out_flat = torch.full((3, flat), base_val, device=device, dtype=torch.float32)

    out_flat[0, has] = 255.0 * pol_last[has]
    out_flat[2, has] = 255.0 * (1.0 - pol_last[has])
    out_flat[1, has] = 255.0 * (1.0 - t_norm_last)

    out = out_flat.view(3, H, W)
    if normalize:
        out = out / 255.0

    return out

@torch.no_grad()
def gpu_polarity_frame_2ch(
    x_np: np.ndarray, y_np: np.ndarray, p_np: np.ndarray,
    H: int, W: int, device: torch.device
) -> Tuple[torch.Tensor, int]:
    if x_np.size == 0:
        return torch.zeros((2, H, W), device=device, dtype=torch.float32), 0

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.int64)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.int64)

    # KEY FIX: keep p as-is, then promote safely before >0
    p = torch.from_numpy(p_np).to(device=device)
    if p.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        p_cmp = p.to(torch.int16)   # prevents uint8 255 -> int8 -1 wrap
    else:
        p_cmp = p

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not torch.any(valid):
        return torch.zeros((2, H, W), device=device, dtype=torch.float32), 0

    x = x[valid]; y = y[valid]; p_cmp = p_cmp[valid]
    n_valid = int(x.numel())

    flat = H * W
    lin = y * W + x
    ch = (p_cmp > 0).to(torch.int64)

    idx = lin + ch * flat

    counts = torch.bincount(idx, minlength=2 * flat).to(torch.float32)
    frame = counts.view(2, flat).view(2, H, W)

    return frame, n_valid

@torch.no_grad()
def vit_preprocess_like_dataloader(pol_2hw: torch.Tensor, out_hw=(224, 224)) -> torch.Tensor:
    """
    Match EventGeMData.__getitem__ preprocessing:
      - input: (2,H,W) float counts on GPU
      - output: (1,2,224,224) float in [-1, 1]
    """
    x = pol_2hw.unsqueeze(0)  # (1,2,H,W)
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

    # Robust Norm (98th percentile) over all elements (matches dataloader)
    flat = x.reshape(-1)
    if flat.numel() > 0:
        k = int(0.98 * flat.numel())
        k = max(1, min(k, flat.numel()))
        robust_max, _ = torch.kthvalue(flat, k)
        robust_max = torch.clamp(robust_max, min=1.0)  # matches "if <1e-6 -> 1.0" behavior safely

        x = torch.clamp(x, max=robust_max)
        x = x / robust_max
        x = x * 2.0 - 1.0

    return x

@torch.no_grad()
def gpu_mcts(
    x_np: np.ndarray, y_np: np.ndarray, t_raw_np: np.ndarray, p_np: np.ndarray,
    H: int, W: int, t_ref_raw: int, time_scale: float, windows_sec: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    num_scales = int(windows_sec.numel())
    C = 2 * num_scales
    out = torch.zeros((C, H, W), device=device, dtype=torch.float32)

    if x_np.size == 0:
        return out

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.int64)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.int64)
    t = torch.from_numpy(t_raw_np).to(device=device, dtype=torch.int64)
    p = torch.from_numpy(p_np).to(device=device, dtype=torch.int8)

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not torch.any(valid):
        return out

    x = x[valid]; y = y[valid]; t = t[valid]; p = p[valid]
    lin = y * W + x
    flat = H * W

    t_rel = (t - int(t_ref_raw)).to(torch.float32) * float(time_scale)
    grp = torch.where(p > 0, torch.zeros_like(lin), torch.ones_like(lin))
    idx2 = lin + grp * flat
    neg_inf = torch.tensor(-1e20, device=device, dtype=torch.float32)

    for s_idx in range(num_scales):
        Dt = float(windows_sec[s_idx].item())
        m = t_rel >= -Dt
        if not torch.any(m):
            continue

        idx_m = idx2[m]
        t_m = t_rel[m]

        t_last = neg_inf.expand(2 * flat).clone()
        t_last.scatter_reduce_(0, idx_m, t_m, reduce="amax", include_self=True)

        valid_pix = t_last > -1e10
        dt = (-t_last).clamp(min=0.0, max=Dt)
        vals = torch.zeros_like(t_last)
        vals[valid_pix] = torch.exp(-dt[valid_pix] / Dt)

        vals = vals.view(2, flat).view(2, H, W)
        out[s_idx, :, :] = vals[0]
        out[num_scales + s_idx, :, :] = vals[1]

    return out

# ---------------------------
# Backbone (ViT + GeM)
# ---------------------------
def load_vit_backbone(backbone_ckpt: str, device: torch.device):
    from external.backbone.model.ours_model.ours_model_pretrain import vit_contrastive_patch16_small

    backbone = vit_contrastive_patch16_small(mask_ratio=0.0, in_chans=2, num_classes=512)

    ckpt = torch.load(backbone_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "checkpoint" in ckpt:
        state_dict = ckpt["checkpoint"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        nk = k.replace("encoder_q.", "").replace("module.", "")
        new_state[nk] = v

    msg = backbone.load_state_dict(new_state, strict=False)
    backbone.to(device).eval()
    return backbone

def infer_vit_input_hw(backbone) -> Tuple[int, int]:
    N = int(backbone.pos_embed.shape[1])
    g = int(round(math.sqrt(N)))
    patch = getattr(backbone.patch_embed, "patch_size", 16)
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)
    img = g * patch
    return img, img

# ---------------------------
# SuperEvent crop logic
# ---------------------------
def compute_superevent_crop_offsets(H: int, W: int, config: dict):
    max_factor_required = config["grid_size"]
    if "backbone_config" in config:
        stage_blocks = config["backbone_config"]["num_blocks"]
        patch_size = config["backbone_config"]["stem"]["patch_size"]
        downsample_factor = patch_size * (2 ** (len(stage_blocks) - 1))
        max_factor_required = downsample_factor

        if "attention" in config["backbone_config"]["stage"]:
            max_partition = np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
            max_factor_required *= max_partition

    crop = np.array([H, W]) % max_factor_required
    off_top = int(math.ceil(crop[0] / 2))
    off_bottom = int(math.floor(crop[0] / 2))
    off_left = int(math.ceil(crop[1] / 2))
    off_right = int(math.floor(crop[1] / 2))

    Hc = H - int(crop[0])
    Wc = W - int(crop[1])
    h_end = H - off_bottom if off_bottom > 0 else H
    w_end = W - off_right if off_right > 0 else W
    return off_top, off_left, off_bottom, off_right, h_end, w_end, Hc, Wc


def load_superevent_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    backbone_name = cfg.get("backbone", None)
    if backbone_name is not None:
        candidates = [
            config_path.parent / "backbones" / f"{backbone_name}.yaml",
            config_path.parent.parent / "backbones" / f"{backbone_name}.yaml",
        ]
        bb_path = next((p for p in candidates if p.exists()), None)
        if bb_path is None:
            raise FileNotFoundError(f"SuperEvent backbone='{backbone_name}' not found.")
        with open(bb_path, "r") as f:
            bb_cfg = yaml.safe_load(f)

        cfg.update(bb_cfg)
        if "backbone_config" in cfg and "input_channels" in cfg:
            cfg["backbone_config"]["input_channels"] = cfg["input_channels"]

    if "backbone_config" not in cfg:
        raise KeyError(f"`backbone_config` missing in {config_path}")

    return cfg

def build_superevent_model(config_path: Path, weights_path: Path, device: torch.device):
    from models.super_event import SuperEvent, SuperEventFullRes

    cfg = load_superevent_config(config_path)
    model = SuperEventFullRes(cfg, tracing=False) if cfg.get("pixel_wise_predictions", False) else SuperEvent(cfg, tracing=False)

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and any(k in state for k in ("model", "state_dict")):
        state = state.get("model", state.get("state_dict", state))

    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg

# ==========================
# Depth config / model utilities
# ==========================

def load_and_merge_config(loadmodel, config):
    """
    Load model checkpoint and configuration, merging with command line arguments.
    Returns: (ckpt, config_dict)
    """
    if loadmodel is not None:
        if not os.path.exists(loadmodel):
            raise FileNotFoundError(f"Model checkpoint not found: Please visit https://drive.google.com/drive/folders/15Yfc1cc6FDsjjpjDdb038u0SyrI8itpv to download into the ./eventgem/external/depthanyevent/models/ folder and try again.")
        print(f"[INFO] Loading model from {loadmodel}")
        ckpt = torch.load(loadmodel, map_location='cpu')

        # External config (optional override)
        external_config = {}
        if config is not None:
            with open(config, 'r') as f:
                external_config = json.load(f)

        # Config from checkpoint or model folder
        if 'config' in ckpt:
            config = ckpt['config']
        else:
            model_folder = os.path.dirname(loadmodel)
            config_file = os.path.join(model_folder, 'config.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError("No config file found in model folder and none in checkpoint.")

        # Merge external config (excluding model section)
        for key in external_config:
            if key != 'model':
                config[key] = external_config[key]

        # Update model checkpoint path
        config['model']['checkpoint_path'] = loadmodel

    else:
        ckpt = None
        if config is None:
            raise ValueError("Either --depth-loadmodel or --depth-config must be specified")
        with open(config, 'r') as f:
            config = json.load(f)

    return ckpt, config

# ---------------------------
# Retrieval DB
# ---------------------------
def load_ref_vit_db(ref_feats_path: str, device: torch.device, dtype: torch.dtype):
    ref = torch.load(ref_feats_path, map_location="cpu")
    if isinstance(ref, dict): ref = ref.get("descriptors", ref)
    if not isinstance(ref, torch.Tensor): ref = torch.tensor(ref, dtype=torch.float32)

    ref = ref.to(device=device, dtype=dtype)
    ref = F.normalize(ref, p=2, dim=1)
    return ref


@torch.no_grad()
def retrieve_topk(ref_db: torch.Tensor, q_desc: torch.Tensor, k: int, return_sims: bool = False):
    q = q_desc[0] if q_desc.ndim == 2 else q_desc
    q = q.to(dtype=ref_db.dtype)
    q = F.normalize(q, p=2, dim=0)

    sims = torch.matmul(ref_db, q)  # [N_ref]
    kk = min(int(k), int(sims.numel()))
    top_sim, top_idx = torch.topk(sims, k=kk, largest=True)
    top_dist = (1.0 - top_sim).to(torch.float32)

    if return_sims:
        return top_idx, top_dist, sims
    return top_idx, top_dist

# ---------------------------
# Batch GPU Rerank Helpers
# ---------------------------
class BatchedRefStore:
    def __init__(self, ref_dir: Path, pattern: str, cache_size: int, max_kpts: int):
        self.ref_dir = Path(ref_dir)
        self.pattern = pattern
        self.max_kpts = max_kpts
        self.cache = OrderedDict()
        self.cache_size = cache_size

    def get_batch_tensor(self, indices: List[int], device: torch.device):
        B = len(indices)
        b_kpts, b_desc, b_mask = [], [], []
        
        for idx in indices:
            idx = int(idx)
            if idx in self.cache:
                self.cache.move_to_end(idx)
                k, d = self.cache[idx]
            else:
                path = self.ref_dir / self.pattern.format(idx)
                if path.exists():
                    data = np.load(str(path))
                    k = data['keypoints'].astype(np.float32)
                    d = data['descriptors'].astype(np.float32)
                    if d.ndim == 1: d = d.reshape(1, -1)
                else:
                    k = np.zeros((0,2), dtype=np.float32)
                    d = np.zeros((0,256), dtype=np.float32)
                
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                self.cache[idx] = (k, d)
            
            n = k.shape[0]
            if n > self.max_kpts:
                k, d = k[:self.max_kpts], d[:self.max_kpts]
                m = np.ones(self.max_kpts, dtype=bool)
            else:
                pad_n = self.max_kpts - n
                k = np.pad(k, ((0, pad_n), (0, 0)))
                d = np.pad(d, ((0, pad_n), (0, 0)))
                m = np.concatenate([np.ones(n, dtype=bool), np.zeros(pad_n, dtype=bool)])
            
            b_kpts.append(k); b_desc.append(d); b_mask.append(m)

        t_kpts = torch.from_numpy(np.stack(b_kpts)).to(device, non_blocking=True)
        t_desc = torch.from_numpy(np.stack(b_desc)).to(device, non_blocking=True)
        t_mask = torch.from_numpy(np.stack(b_mask)).to(device, non_blocking=True)
        t_desc = F.normalize(t_desc, p=2, dim=2)
        
        return t_kpts, t_desc, t_mask
    
@torch.no_grad()
def vit_gem_descriptor(backbone, x_bchw: torch.Tensor) -> torch.Tensor:
    t = backbone.patch_embed(x_bchw)
    t = t + backbone.pos_embed
    t = torch.cat((backbone.tokens.expand(t.shape[0], -1, -1), t), dim=1)

    for blk in backbone.blocks:
        t = blk(t)
    t = backbone.norm(t)

    patch_tokens = t[:, 2:, :]
    B, N, C = patch_tokens.shape
    g = int(round(math.sqrt(N)))
    patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, g, g)

    p = 5.0
    gem = F.avg_pool2d((patch_tokens.clamp(min=1e-6)).pow(p), (g, g)).pow(1.0 / p)
    return gem.squeeze(-1).squeeze(-1)

# ---------------------------
# Image Descriptors
# ---------------------------
@torch.no_grad()
def sample_descriptors_at_kpts(keypoints_yx: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
    if keypoints_yx.numel() == 0:
        return torch.empty((0, descriptors.shape[1]), device=descriptors.device, dtype=descriptors.dtype)

    Hc, Wc = descriptors.shape[-2], descriptors.shape[-1]
    k_xy = keypoints_yx[:, [1, 0]]
    grid = torch.empty((1, 1, k_xy.shape[0], 2), device=descriptors.device, dtype=descriptors.dtype)
    grid[0, 0, :, 0] = 2.0 * k_xy[:, 0] / (Wc - 1) - 1.0
    grid[0, 0, :, 1] = 2.0 * k_xy[:, 1] / (Hc - 1) - 1.0

    samp = F.grid_sample(descriptors, grid, mode="bilinear", align_corners=True)
    samp = samp[0, :, 0, :].t()
    samp = F.normalize(samp, p=2, dim=1)
    return samp


@torch.no_grad()
def batched_ransac_rerank(
    q_kpts: torch.Tensor,       # [Nq, 2]
    q_desc: torch.Tensor,       # [Nq, D]
    ref_store: BatchedRefStore,
    cand_indices: np.ndarray,   # [B]
    max_matches: int = 512,
    ratio_thresh: float = 0.8,
    iterations: int = 64,       # RANSAC iterations per candidate
    thresh_px: float = 5.0      # Inlier threshold in pixels
):
    """
    Fully Vectorized GPU RANSAC using Robust DLT.
    """
    B = len(cand_indices)
    if B == 0:
        return np.zeros(0, dtype=np.int32)
    
    device = q_kpts.device
    # Use DLT (SVD-based) instead of get_perspective_transform (Solve-based) to prevent crashes
    from kornia.geometry.homography import find_homography_dlt

    # --- 1. Batch Match ---
    # Fetch Ref: [B, Nr, D]
    r_kpts, r_desc, r_mask = ref_store.get_batch_tensor(cand_indices, device)
    
    # Cosine Sim: [B, Nq, Nr]
    sim = torch.einsum('id,bjd->bij', q_desc, r_desc) 
    sim.masked_fill_(~r_mask.unsqueeze(1), -2.0)
    
    # Ratio Test
    top_val, top_idx = torch.topk(sim, k=2, dim=2) 
    pass_ratio = (1.0 - top_val[:, :, 0]) < (ratio_thresh * (1.0 - top_val[:, :, 1]))
    
    # Gather Top Matches
    s1_masked = top_val[:, :, 0].clone()
    s1_masked[~pass_ratio] = -10.0

    # Select best 'max_matches' per candidate
    _, best_match_indices = torch.topk(s1_masked, k=max_matches, dim=1) 
    
    # Gather Points: [B, M, 2]
    src_pts = q_kpts[best_match_indices.view(-1)].view(B, max_matches, 2)
    ref_match_indices = torch.gather(top_idx[:,:,0], 1, best_match_indices) 
    dst_pts = torch.gather(r_kpts, 1, ref_match_indices.unsqueeze(-1).expand(-1, -1, 2))

# --- 1. Identify Valid Matches (WITHIN THE 512 LIMIT) ---
    # pass_ratio was calculated on the full set, 
    # but we only kept 'max_matches' (512) for src_pts/dst_pts.
    # We must slice pass_ratio to match.
    gathered_pass = torch.gather(pass_ratio, 1, best_match_indices) # [B, 512]
    num_valid_per_cand = gathered_pass.sum(dim=1) # [B]
    
    # Sort the 512 points so valid ones (1s) come before invalid ones (0s)
    _, sort_idx = torch.sort(gathered_pass.float(), dim=1, descending=True)
    
    # Now gather from the ALREADY-CAPPED src_pts and dst_pts
    src_pts = torch.gather(src_pts, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
    dst_pts = torch.gather(dst_pts, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
    
    # --- 2. Vectorized Range-Bound Sampling ---
    rand_floats = torch.rand((B, iterations, 4), device=device)
    
    # Scale by num_valid so we only pick from the "packed" valid matches at the front
    # We clamp at 4 because RANSAC needs at least 4 points to run the DLT math
    sampling_limit = num_valid_per_cand.view(B, 1, 1).clamp(min=4)
    rand_idx = (rand_floats * sampling_limit).long()
    
    # --- 3. Global Indexing (Same as before but now restricted) ---
    batch_offsets = torch.arange(B, device=device).view(B, 1, 1) * max_matches
    global_rand_idx = (rand_idx + batch_offsets).view(-1)
    
    # ... rest of your DLT and inlier counting logic ...
    
    # Gather 4-point sets
    # Expand src_pts to [B, iter, M, 2] is too big memory-wise?
    # Optimization: indexing directly
    # Helper: offset batch indices
    # flat_rand_idx: [B, iter, 4] -> values in 0..M-1
    # We need global indices into flattened [B*M, 2] array
    
    # batch_offsets = torch.arange(B, device=device).view(B, 1, 1) * max_matches
    # global_rand_idx = (rand_idx + batch_offsets).view(-1) # [B*iter*4]
    
    src_flat = src_pts.view(-1, 2) # [B*M, 2]
    dst_flat = dst_pts.view(-1, 2)
    
    ps_src = src_flat[global_rand_idx].view(B, iterations, 4, 2)
    ps_dst = dst_flat[global_rand_idx].view(B, iterations, 4, 2)
    
    # Flatten for DLT
    ps_src_k = ps_src.reshape(-1, 4, 2)
    ps_dst_k = ps_dst.reshape(-1, 4, 2)
    
    # Compute Homographies using DLT (Robust SVD)
    # find_homography_dlt never crashes on singular inputs
    H = find_homography_dlt(ps_src_k, ps_dst_k, weights=None)
    
    # --- 3. Verify Inliers ---
    # Verify all points: [B, 1, M, 2]
    
    # H: [B*iter, 3, 3] -> [B, iter, 3, 3]
    H_view = H.view(B, iterations, 3, 3)
    
    # Prepare Src Points Homogeneous: [B, 1, 3, M]
    ones = torch.ones((B, 1, max_matches, 1), device=device)
    src_h = torch.cat([src_pts.unsqueeze(1), ones], dim=3) 
    src_h_t = src_h.transpose(2, 3) # [B, 1, 3, M]
    
    # Transform: [B, iter, 3, 3] @ [B, 1, 3, M] -> [B, iter, 3, M]
    src_warped_h = H_view @ src_h_t
    
    # Normalize
    w = src_warped_h[:, :, 2:3, :] + 1e-7
    src_warped = src_warped_h[:, :, 0:2, :] / w # [B, iter, 2, M]
    
    # Distances
    # dst_pts: [B, M, 2] -> [B, 1, 2, M]
    dst_t = dst_pts.unsqueeze(1).transpose(2, 3)
    diff = src_warped - dst_t
    dist_sq = diff.pow(2).sum(dim=2) # [B, iter, M]
    
    # Count Inliers
    # Must be geometrically close AND originally valid
    is_inlier = dist_sq < (thresh_px**2)
    is_valid_point = gathered_pass.unsqueeze(1) # [B, 1, M]
    
    final_inliers = is_inlier & is_valid_point
    
    # Best iteration
    count = final_inliers.sum(dim=2) # [B, iter]
    best_counts, _ = count.max(dim=1)
    
    return best_counts.cpu().numpy().astype(np.int32)

class _DepthLRUCache:
    """Tiny LRU cache for *downsampled* reference depth maps."""
    def __init__(self, max_items: int = 512):
        from collections import OrderedDict
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key: int):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: int, value: np.ndarray):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)

def _load_depth_map(depth_dir: Path, idx: int, pattern: str, index_offset: int = 0, down_hw=None):
    """
    Load a single depth PNG (expected 16-bit), return float32.
    Optionally downsamples to down_hw=(H,W) for speed.
    """
    path = depth_dir / pattern.format(idx + index_offset)
    if not path.exists():
        return None
    D = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if D is None:
        return None
    if D.ndim == 3:
        D = D[..., 0]
    D = D.astype(np.float32)
    if down_hw is not None:
        Ht, Wt = int(down_hw[0]), int(down_hw[1])
        D = cv2.resize(D, (Wt, Ht), interpolation=cv2.INTER_AREA)
    return D

def rerank_depth_single_query(
    qD: np.ndarray,
    base_dist: np.ndarray,                      # shape (R,)
    ref_depth_dir: str | Path,
    top_k: int = 50,
    topk_idx: Optional[Sequence[int]] = None,   # optional precomputed topK ref ids
    depth_pattern: str = "depth_{:06d}.png",
    depth_index_offset: int = 0,
    depth_down_hw: Optional[Tuple[int, int]] = (28,28),  # (H,W)
    depth_weight: float = 0.15,
    tau: float = 0.3,
    err_threshold: float = 0.7,                 # only boost if err < threshold
    cache: Optional[_DepthLRUCache] = None,
    load_fn: Callable[..., Optional[np.ndarray]] = _load_depth_map,
) -> np.ndarray:
    """
    Returns a reranked distance vector for ONE query.

    Logic:
      - pick topK refs (by smallest base distance)
      - load ONLY those refs' depth maps
      - compute depth errors for them
      - convert to similarity boost: sim = exp(-err/tau)
      - apply boost only when err < err_threshold:
            new_dist[ref] = base_dist[ref] - depth_weight * sim
    """
    base_dist = np.asarray(base_dist, dtype=np.float32)
    R = base_dist.shape[0]
    dist_fn = _compute_robust_depth_distance
    qD = qD.squeeze(0).squeeze(0).cpu().numpy()

    ref_depth_dir = Path(ref_depth_dir)
    if cache is None:
        cache = _DepthLRUCache(max_items=0)  # disabled unless you pass one

    # Choose topK refs (or use provided)
    K = int(min(max(top_k, 1), R))
    if topk_idx is None:
        part = np.argpartition(base_dist, kth=K - 1)[:K].astype(np.int32, copy=False)
        refs = part[np.argsort(base_dist[part])].astype(np.int32, copy=False)
    else:
        refs = np.asarray(topk_idx, dtype=np.int32)
        if refs.size > K:
            refs = refs[:K]

    # Start with original distances
    new_dist = base_dist.copy()

    if qD is None or refs.size == 0:
        return new_dist

    # reshape qd to downsampled size if needed
    if depth_down_hw is not None:
        qD = cv2.resize(qD, (depth_down_hw[1], depth_down_hw[0]), interpolation=cv2.INTER_AREA)

    # Compute depth errors for topK only
    errs = np.full((refs.size,), np.inf, dtype=np.float32)

    for j, r_idx in enumerate(refs):
        r_idx_int = int(r_idx)

        rD = cache.get(r_idx_int)
        if rD is None:
            rD = load_fn(ref_depth_dir, r_idx_int, depth_pattern, depth_index_offset, depth_down_hw)
            if rD is not None:
                cache.put(r_idx_int, rD)

        if rD is None:
            continue

        errs[j] = float(dist_fn(rD, qD))

    # Convert to similarity and apply gated boost
    sims = np.exp(-errs / float(tau)).astype(np.float32)
    mask_good = np.isfinite(errs) & (errs < float(err_threshold))

    if np.any(mask_good):
        good_refs = refs[mask_good]
        new_dist[good_refs] = base_dist[good_refs] - (float(depth_weight) * sims[mask_good])

    return new_dist

def _compute_robust_depth_distance(Rr: np.ndarray, Qr: np.ndarray) -> float:
    """
    Computes a structural distance between two depth maps.
    Uses SSIM on normalized, masked depth to handle viewpoint shifts.
    """
    if Rr is None or Qr is None:
        return np.inf

    # 1. Create a mask of valid pixels (non-zero in both)
    mask = (Rr > 0) & (Qr > 0)
    
    # Safety check: if overlap is non-existent, it's a mismatch
    if np.sum(mask) < 100: 
        return 1.0

    # 2. Normalize intensity to [0, 1] based on valid percentiles
    def normalize_depth(D, m):
        vals = D[m]
        if len(vals) == 0: return D
        v_min, v_max = np.percentile(vals, [1, 99])
        # Avoid division by zero
        denom = v_max - v_min if v_max > v_min else 1.0
        return np.clip((D - v_min) / denom, 0, 1)

    Rr_n = normalize_depth(Rr, mask)
    Qr_n = normalize_depth(Qr, mask)

    # 3. Compute SSIM
    # mssim: the global average (scalar)
    # ssim_map: the per-pixel similarity map (array)
    mssim, ssim_map = ssim(Rr_n, Qr_n, full=True, data_range=1.0, win_size=7)
    
    # 4. Correctly index the MAP, not the scalar
    valid_scores = ssim_map[mask]
    
    if valid_scores.size == 0:
        return 1.0
        
    avg_structural_sim = np.mean(valid_scores)
    
    # Return 1.0 (worst) to 0.0 (perfect structural match)
    return 1.0 - float(avg_structural_sim)
    
# ---------------------------
# Stats formatting
# ---------------------------
def summarize_ms(arr: np.ndarray, name: str) -> str:
    if arr.size == 0:
        return f"{name}: (no data)"
    return f"{name} ms | mean {arr.mean():.2f}  med {np.median(arr):.2f}  p95 {np.percentile(arr,95):.2f}  max {arr.max():.2f}"

def build_reranked_column_from_sims(
    sims_t: torch.Tensor,
    cand_ids: np.ndarray,
    inlier_counts: np.ndarray,
    inlier_weight: float,
) -> np.ndarray:
    """
    Build a dense reranked distance column for ONE query.

    - sims_t: torch tensor [N_ref] (cosine sims, higher is better) or any 1D similarity vector.
    - cand_ids: np int array [B] candidate reference indices (e.g., top_idx_t from retrieve_topk).
    - inlier_counts: np int/float array [B] inlier counts per candidate (same order as cand_ids).
    - inlier_weight: float, applied as dist -= inlier_weight * inlier_count

    Returns:
      reranked_dist_col: np.ndarray [N_ref] float32, where non-candidates keep base dist (1-sim),
      and candidates get adjusted distances.
    """
    # base distance column: lower is better
    sims_np = sims_t.detach().float().cpu().numpy().reshape(-1)  # [N_ref]
    base_dist = (1.0 - sims_np).astype(np.float32, copy=False)   # [N_ref]

    # no candidates => identical to base
    if cand_ids is None or len(cand_ids) == 0:
        return base_dist

    cand_ids = np.asarray(cand_ids, dtype=np.int64).reshape(-1)
    inlier_counts = np.asarray(inlier_counts, dtype=np.float32).reshape(-1)

    # defensive: align lengths
    B = min(cand_ids.size, inlier_counts.size)
    cand_ids = cand_ids[:B]
    inlier_counts = inlier_counts[:B]

    # adjusted distances for candidates
    adj = (inlier_weight * inlier_counts).astype(np.float32, copy=False)
    base_dist[cand_ids] = base_dist[cand_ids] - adj
    return base_dist

def recallAtK(S, GT, K=1):
    """
    Calculates the recall@K for a given similarity matrix S and ground truth matrix GT.
    Note that this method does not support GTsoft - instead, please directly provide
    the dilated ground truth matrix as GT.

    The matrices S and GT are two-dimensional and should all have the same shape.
    The matrix GT should be binary, where the entries are only zeros or ones.
    The matrix S should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The integer K>=1 defines the number of matching candidates that are selected and
    that must contain an actually matching image pair.
    """

    assert (S.shape == GT.shape),"S and GT must have the same shape"
    assert (S.ndim == 2),"S and GT must be two-dimensional"
    assert (K >= 1),"K must be >=1"

    # ensure logical datatype in GT
    GT = GT.astype('bool')

    # discard all query images without an actually matching database image
    j = GT.sum(0) > 0 # columns with matches
    S = S[:,j] # select columns with a match
    GT = GT[:,j] # select columns with a match

    # select K highest similarities
    i = S.argsort(0)[-K:,:]
    j = np.tile(np.arange(i.shape[1]), [K, 1])
    GT = GT[i, j]

    # recall@K
    RatK = np.sum(GT.sum(0) > 0) / GT.shape[1]

    return RatK

def build_reranked_column_from_topdists(
    N_ref: int,
    sims_t: torch.Tensor,
    cand_ids: np.ndarray,
    cand_base_dist: np.ndarray,
    inlier_counts: np.ndarray,
    inlier_weight: float,
    *,
    fill_mode: str = "base",   # "base" uses (1-sims) for non-cands, "inf" sets non-cands to +inf
) -> np.ndarray:
    """
    Alternative: if you already have cand_base_dist = top_dist_t (1 - top_sim) for the shortlist,
    you can compute reranked values for shortlist and fill the rest either with base distances
    from sims_t or +inf.

    Returns:
      reranked_dist_col: np.ndarray [N_ref] float32
    """
    cand_ids = np.asarray(cand_ids, dtype=np.int64).reshape(-1)
    cand_base_dist = np.asarray(cand_base_dist, dtype=np.float32).reshape(-1)
    inlier_counts = np.asarray(inlier_counts, dtype=np.float32).reshape(-1)

    B = min(cand_ids.size, cand_base_dist.size, inlier_counts.size)
    cand_ids = cand_ids[:B]
    cand_base_dist = cand_base_dist[:B]
    inlier_counts = inlier_counts[:B]

    if fill_mode == "inf":
        col = np.full((N_ref,), np.inf, dtype=np.float32)
    elif fill_mode == "base":
        sims_np = sims_t.detach().float().cpu().numpy().reshape(-1)
        col = (1.0 - sims_np).astype(np.float32, copy=False)
        if col.size != N_ref:
            raise ValueError(f"sims_t has {col.size} refs but N_ref={N_ref}")
    else:
        raise ValueError("fill_mode must be 'base' or 'inf'")

    col[cand_ids] = cand_base_dist - (inlier_weight * inlier_counts)
    return col

# ---------------------------
# SuperEvent Helpers
# ---------------------------
def preload_ref_descs(
    ref_kp_dir: Union[str, Path],
    ref_kp_pattern: str = "mcts_{:05d}.feat.npz",
    desc_key: str = "descriptors",
    dtype=np.float32,
) -> List[Optional[np.ndarray]]:
    """
    Load all ref descriptors from `ref_kp_dir` matching `ref_kp_pattern`.

    Returns:
        ref_descs: list where ref_descs[i] is the descriptor array for index i,
                   or None if that index/file is missing.
                   Each descriptor array is shape (N, D).
    """
    ref_kp_dir = Path(ref_kp_dir)

    # Turn format pattern into a glob, e.g. "mcts_{:05d}.feat.npz" -> "mcts_*.feat.npz"
    glob_pat = re.sub(r"\{[^}]*\}", "*", ref_kp_pattern)
    files = list(ref_kp_dir.glob(glob_pat))
    if not files:
        return []

    # Build a regex to extract the numeric index from filename based on the pattern
    # e.g. "mcts_{:05d}.feat.npz" -> r"^mcts_(\d+)\.feat\.npz$"
    regex_pat = "^" + re.escape(ref_kp_pattern) + "$"
    regex_pat = re.sub(r"\\\{[^}]*\\\}", r"(\\d+)", regex_pat)
    rx = re.compile(regex_pat)

    indexed = []
    for p in files:
        m = rx.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        indexed.append((idx, p))

    if not indexed:
        return []

    indexed.sort(key=lambda t: t[0])
    max_idx = indexed[-1][0]
    ref_descs: List[Optional[np.ndarray]] = [None] * (max_idx + 1)
    from tqdm import tqdm

    for idx, path in tqdm(indexed, desc="Preloading ref descriptors", unit="file"):
        with np.load(str(path)) as data:
            if desc_key not in data:
                raise KeyError(f"'{desc_key}' not found in {path.name}. Keys: {list(data.keys())}")
            desc = data[desc_key]

        # Ensure 2D (N, D)
        desc = np.asarray(desc)
        if desc.ndim == 1:
            desc = desc.reshape(1, -1)
        elif desc.ndim != 2:
            raise ValueError(f"Expected descriptors to be 1D or 2D, got shape {desc.shape} in {path.name}")

        if dtype is not None:
            desc = desc.astype(dtype, copy=False)

        ref_descs[idx] = desc

    return ref_descs

def bruteforce(query_desc, ref_descs):

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 

    distMat = np.zeros((len(query_desc), len(ref_descs)))
    for j, r_desc in enumerate(ref_descs):
        q_np = query_desc.detach().cpu().numpy()
        matches = bf.match(q_np, r_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) == 0:
            distMat[0, j] = 1.0
            continue
        avg_dist = np.mean([m.distance for m in matches])
        distMat[0, j] = avg_dist

    return distMat