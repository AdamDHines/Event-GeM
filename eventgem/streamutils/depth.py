#!/usr/bin/env python3
"""
Event-based Depth Prediction Script (Tencode + DAv2/RecDAv2)

- Loads a DAv2 / RecDAv2 depth model from checkpoint/config
- Reads event-based .hdf5 files
- Converts events to Tencode representation in fixed time windows
- Runs depth inference and saves depth images as PNGs

Assumes:
- HDF5 layout: group "events" with datasets "x", "y", "t", "p"
- Timestamps in nanoseconds (default time_scale=1e-9)
"""

from __future__ import print_function
import argparse
from html import parser
import os
from pathlib import Path
import json
import random

import h5py
import numpy as np
import cv2
import torch
from torch import autocast
from tqdm import tqdm
import cmapy
from PIL import Image

from eventgem.external.depthanyevent.models import fetch_model  # from the DepthAnyEvent repo


# ==========================
# HDF5 + Tencode utilities
# ==========================


EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")


def _find_first_dataset(g: h5py.Group, candidates: tuple[str, ...], logical_name: str):
    """
    Return the first dataset in `g` (or any of its subgroups) whose name matches
    one of `candidates`.

    This handles layouts like:
      /x, /y, /t, /p
    as well as:
      /columns/x, /columns/y, /columns/t, /columns/p
    or deeper trees under 'events'.
    """

    # 1) Try direct children first (original behaviour)
    for key in candidates:
        if key in g:
            obj = g[key]
            if isinstance(obj, h5py.Dataset):
                return obj

    # 2) If not found, search recursively in sub-groups
    for name, obj in g.items():
        if isinstance(obj, h5py.Group):
            try:
                return _find_first_dataset(obj, candidates, logical_name)
            except RuntimeError:
                # No match in this subgroup, keep looking
                continue

    raise RuntimeError(
        f"Could not find {logical_name} dataset under group '{g.name}'. "
        f"Tried names: {', '.join(candidates)}"
    )


def find_event_datasets(f: h5py.File):
    """
    Return (x, y, t, p) datasets from an HDF5 file.

    Looks in group 'events' if it exists (and is a group), otherwise
    falls back to the file root. For each component, tries multiple
    possible key names defined in EVENT_*_KEYS.
    """
    if "events" in f and isinstance(f["events"], h5py.Group):
        g = f["events"]
    else:
        # Some datasets store x/y/t/p directly at the root
        g = f

    x_ds = _find_first_dataset(g, EVENT_X_KEYS, "x")
    y_ds = _find_first_dataset(g, EVENT_Y_KEYS, "y")
    t_ds = _find_first_dataset(g, EVENT_T_KEYS, "t")
    p_ds = _find_first_dataset(g, EVENT_P_KEYS, "p")

    return x_ds, y_ds, t_ds, p_ds


def infer_resolution_stream(x_dset, y_dset, chunk_size: int = 200_000):
    """
    Infer sensor resolution by streaming through x/y once.
    """
    N = len(x_dset)
    H_max, W_max = 0, 0

    if N == 0:
        raise ValueError("No events in dataset; cannot infer resolution.")

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        xs = x_dset[start:end][:]
        ys = y_dset[start:end][:]

        if xs.size == 0:
            continue

        W_max = max(W_max, int(xs.max()) + 1)
        H_max = max(H_max, int(ys.max()) + 1)

    print(f"[INFO] Inferred resolution HxW = {H_max}x{W_max}")
    return H_max, W_max


def tencode_numpy(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    white_frame: bool = False,
    normalize: bool = True,
):
    """
    NumPy implementation of the Tencode mapping used in DepthAnyEvent/event_representation.

    - x, y, t, p are 1D arrays for a single temporal chunk (window)
    - height, width: frame size
    - white_frame: if True, initialise with white (255) instead of black
    - normalize: if True, output in [0,1]; otherwise [0,255]
    """
    assert x.ndim == y.ndim == t.ndim == p.ndim == 1
    n = x.shape[0]
    base_val = 255.0 if white_frame else 0.0

    if n == 0:
        frame = np.full((3, height, width), base_val, dtype=np.float32)
        return frame / 255.0 if normalize else frame

    # Convert to correct dtypes
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    t = t.astype(np.float64)
    p = p.astype(np.float64)

    # Sort by time so "last event wins" at each pixel
    order = np.argsort(t)
    x = x[order]
    y = y[order]
    t = t[order]
    p = p[order]

    # Polarity to {0,1}
    if p.min() < 0:
        pol = (p > 0).astype(np.float32)
    else:
        pol = (p > 0).astype(np.float32)

    # Normalise time to [0,1] within this window
    if t[-1] != t[0]:
        t_norm = (t - t[0]) / (t[-1] - t[0])
    else:
        t_norm = np.zeros_like(t, dtype=np.float32)

    tencode = np.full((3, height, width), base_val, dtype=np.float32)

    # Valid indices
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return tencode / 255.0 if normalize else tencode

    xv = x[valid]
    yv = y[valid]
    polv = pol[valid]
    tn = t_norm[valid]

    # R: 255 for positive polarity, else 0
    tencode[0, yv, xv] = 255.0 * polv
    # G: 255 * (1 - t_norm) -> newest events darker
    tencode[1, yv, xv] = 255.0 * (1.0 - tn)
    # B: 255 for negative polarity, else 0
    tencode[2, yv, xv] = 255.0 * (1.0 - polv)

    if normalize:
        tencode = tencode / 255.0

    return tencode


def save_depth_png(depth_map: np.ndarray,
                   out_path: Path,
                   clip_distance: float,
                   gamma: float):
    """
    depth_map: linear depth in meters (2D array) after conversion.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True) 

    depth = np.nan_to_num(depth_map, nan=0.0, posinf=clip_distance, neginf=0.0)
    depth = np.clip(depth, 0.0, clip_distance)

    # normalize and apply gamma
    depth_norm = depth / clip_distance if clip_distance > 0 else depth
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_norm = depth_norm ** gamma

    img_gray = (depth_norm * 255.0).astype(np.uint8)
    # use the same magma colormap as the repo
    img_color = cv2.applyColorMap(img_gray, cmapy.cmap('magma'))

    ok = cv2.imwrite(str(out_path), img_color)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out_path}")

def binary_frame_numpy(x: np.ndarray, y: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Binary on/off frame for a window.
    Output: (H,W) uint8, 255 where any event occurred, else 0.
    """
    frame = np.zeros((height, width), dtype=np.uint8)
    if x.size == 0:
        return frame

    x = x.astype(np.int64, copy=False)
    y = y.astype(np.int64, copy=False)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return frame

    frame[y[valid], x[valid]] = 255
    return frame


def polarity_split_frame_numpy(x: np.ndarray, y: np.ndarray, p: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Polarity split frame for a window.
    Output: (H,W,3) uint8 RGB, R=pos on/off, B=neg on/off.
    If a pixel has both polarities within the window, it becomes magenta (R+B).
    """
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    if x.size == 0:
        return rgb

    x = x.astype(np.int64, copy=False)
    y = y.astype(np.int64, copy=False)
    p = p.astype(np.float32, copy=False)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return rgb

    xv = x[valid]
    yv = y[valid]
    pv = p[valid]

    pos = pv > 0
    neg = ~pos

    if np.any(pos):
        rgb[yv[pos], xv[pos], 0] = 255  # R
    if np.any(neg):
        rgb[yv[neg], xv[neg], 2] = 255  # B

    return rgb


def save_gray_rep(frame_hw_u8: np.ndarray, parent_outdir: Path, subdir: str, frame_name: str):
    """
    Saves (H,W) uint8 into parent_outdir/<subdir>/<frame_name>.png
    """
    out_dir = Path(parent_outdir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_hw_u8, mode="L").save(out_dir / f"{frame_name}.png")


def save_rgb_rep(frame_hwc_u8: np.ndarray, parent_outdir: Path, subdir: str, frame_name: str):
    """
    Saves (H,W,3) uint8 into parent_outdir/<subdir>/<frame_name>.png
    """
    out_dir = Path(parent_outdir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_hwc_u8, mode="RGB").save(out_dir / f"{frame_name}.png")


def save_tencode_outputs(
    tencode_chw: np.ndarray,        # (3,H,W) float in [0,1] (as you generate)
    parent_outdir: Path,
    frame_name: str,                # e.g. "depth_000123"
    normalize: bool = True,
):
    """
    Saves Tencode PNG into:
      parent_outdir/tencode/<frame_name>.png
    """
    parent_outdir = Path(parent_outdir)
    tencode_dir = parent_outdir / "tencode"
    tencode_dir.mkdir(parents=True, exist_ok=True)

    ten = tencode_chw
    if normalize:
        img = np.clip(ten * 255.0, 0, 255).astype(np.uint8)
    else:
        img = np.clip(ten, 0, 255).astype(np.uint8)

    img = np.transpose(img, (1, 2, 0))  # (3,H,W) -> (H,W,3), RGB
    Image.fromarray(img, mode="RGB").save(tencode_dir / f"{frame_name}.png")


def save_depth_outputs(
    depth_m: np.ndarray,
    parent_outdir: Path,
    frame_name: str,              # e.g. "depth_000123"
    clip_distance: float,
    gamma: float,
    save_npy: bool = True,
    save_raw16: bool = True,
    save_viz: bool = True,
):
    """
    Saves depth outputs into:
      parent_outdir/
        npy/    depth_XXXXXX.npy
        raw16/  depth_XXXXXX.png        (uint16 single-channel)
        viz/    depth_XXXXXX.png        (uint8 magma)
    """
    parent_outdir = Path(parent_outdir)

    if save_raw16: parent_outdir.mkdir(parents=True, exist_ok=True)

    depth = np.nan_to_num(depth_m, nan=0.0, posinf=clip_distance, neginf=0.0)
    depth = np.clip(depth, 0.0, clip_distance).astype(np.float32)

    if save_raw16:
        if clip_distance <= 0:
            raise ValueError("clip_distance must be > 0 for raw16 encoding")
        depth_u16 = np.round((depth / clip_distance) * 65535.0).astype(np.uint16)
        ok = cv2.imwrite(str(parent_outdir / f"{frame_name}.png"), depth_u16)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {parent_outdir / f'{frame_name}.png'}")


# ==========================
# Config / model utilities
# ==========================

def load_and_merge_config(args):
    """
    Load model checkpoint and configuration, merging with command line arguments.
    Returns: (ckpt, config_dict)
    """
    if args.depth_model is not None:
        print(f"[INFO] Loading model from {args.depth_model}")
        ckpt = torch.load(args.depth_model, map_location='cpu')

        # External config (optional override)
        external_config = {}
        if args.depth_config is not None:
            with open(args.depth_config, 'r') as f:
                external_config = json.load(f)

        # Config from checkpoint or model folder
        if 'config' in ckpt:
            config = ckpt['config']
        else:
            model_folder = os.path.dirname(args.depth_model)
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
        config['model']['checkpoint_path'] = args.depth_model

    else:
        ckpt = None
        if args.config is None:
            raise ValueError("Either --loadmodel or --config must be specified")
        with open(args.config, 'r') as f:
            config = json.load(f)

    return ckpt, config


def setup_device_and_seeds(args, seed=42):
    """
    Setup device (CPU/CUDA) and random seeds for reproducibility.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    device = torch.device('cuda:0')

    print(f"[INFO] Using device {device}")
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
    return device, autocast_device

def convert_pred_for_vis(prediction: np.ndarray,
                         use_logdepth: bool,
                         clip_distance: float,
                         reg_factor: float = 3.70378) -> np.ndarray:
    """
    Convert network prediction to linear depth in meters for visualisation.

    - If use_logdepth: treat prediction as log-depth in [0,1] and convert
      like evaluation.prepare_prediction_data (but without any GT).
    - Otherwise: assume prediction is already linear depth.
    """
    pred = prediction.astype(np.float32)

    if use_logdepth:
        # from prepare_prediction_data
        # 1) log-depth -> normalized linear depth
        pred = np.exp(reg_factor * (pred - 1.0))

        # 2) normalize by its own max (scale-invariant)
        # valid = pred[~np.isnan(pred)]
        # max_val = valid.max() if valid.size > 0 else 0.0
        # if max_val > 0:
        #     pred = pred / max_val

        # 3) scale to clip_distance (meters)
        pred = pred * clip_distance

    else:
        # treat as already-linear depth
        pred = np.clip(pred, 0.0, clip_distance)

    return pred

# ==========================
# Core processing
# ==========================

@torch.no_grad()
def process_depth(
    model,
    model_name: str,
    device: torch.device,
    autocast_device: str,
    ref_dir,
    qry_dir,
    args,
):
    dirs = [ref_dir, qry_dir]
    ref_out = f"{args.depth_out}/{args.dataset}/{args.query}-{args.dt_ms}/depth"
    qry_out = f"{args.depth_out}/{args.dataset}/{args.query}-{args.dt_ms}/depth"
    # make the dirs if they don't exist
    os.makedirs(ref_out, exist_ok=True)
    os.makedirs(qry_out, exist_ok=True)
    out_dirs = [ref_out, qry_out]
    prev_states = None 
    for idx, d in enumerate(dirs):
        # get the file list and sort it
        file_list = sorted(Path(d).glob("*.png"))
        for file_idx, file in enumerate(file_list):
            # load tencode image
            tencode_img = Image.open(file).convert("RGB")
            # convert img to numpy
            tencode_img = np.array(tencode_img).astype(np.float32) / 255.0  # (H,W,3) in [0,1]
            tencode_img = np.transpose(tencode_img, (2, 0, 1))  # (3,H,W)
            # To torch
            ev_tensor = torch.from_numpy(tencode_img).float().unsqueeze(0).to(device)  # (1,3,H,W)

            # Inference (no grad, via decorator)
            with autocast(autocast_device, enabled=False):
                if model_name == 'DAv2':
                    pred = model.infer_image(ev_tensor)  # (1,1,H,W)
                elif model_name == 'RecDAv2':
                    pred, prev_states = model.infer_image(ev_tensor, prev_states=prev_states)
                else:
                    raise ValueError(f"Model {model_name} not implemented in this script.")

            # raw network output (log-depth or linear)
            pred_np_raw = pred.squeeze().detach().cpu().numpy()

            # convert to linear depth in meters, respecting use_logdepth/reg_factor/clip_distance
            depth_for_vis = convert_pred_for_vis(
                pred_np_raw,
                use_logdepth=args.use_logdepth,
                clip_distance=args.clip_distance,
                reg_factor=args.reg_factor,
            )

            save_depth_outputs(
                depth_for_vis,              # (this should be linear depth in meters)
                parent_outdir=out_dirs[idx],  # the per-HDF5 directory you already make
                frame_name=f"depth_{file_idx:05d}",  # zero-padded frame index
                clip_distance=args.clip_distance,
                gamma=args.gamma,
                save_npy=False,
                save_raw16=True,
                save_viz=False,
            )

            # Free per-frame tensors explicitly (belt-and-braces)
            del ev_tensor, pred
            torch.cuda.empty_cache() if device.type == 'cuda' else None