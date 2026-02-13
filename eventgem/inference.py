#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path
from typing import Optional, Tuple, List
from collections import OrderedDict

import os
import sys
import yaml
import h5py
import numpy as np
import torch
import torch.nn.functional as F
# Kornia is required for the Batched GPU RANSAC
try:
    from kornia.geometry.transform import get_perspective_transform
except ImportError:
    print("Error: Kornia is required. Install with: pip install kornia")
    sys.exit(1)

THIS_DIR = Path(__file__).resolve().parent
BACKBONE_ROOT = THIS_DIR / "external" / "backbone"
SUPEREVENT_ROOT = THIS_DIR / "external" / "superevent"

# Order matters: backbone's `utils` must win
sys.path.insert(0, str(SUPEREVENT_ROOT))
sys.path.insert(0, str(BACKBONE_ROOT))


# ---------------------------
# HDF5 helpers
# ---------------------------
EVENT_T_KEYS = ("t", "timestamp", "timestamps", "time", "times")
EVENT_X_KEYS = ("x", "x_coordinate", "x_coordinates", "u", "col", "column")
EVENT_Y_KEYS = ("y", "y_coordinate", "y_coordinates", "v", "row")
EVENT_P_KEYS = ("p", "polarity", "polarities", "pol", "polarity_bit", "polarity_bits")

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
# Time helpers
# ---------------------------
def sec_to_raw(t_sec: float, time_scale: float) -> int:
    return int(round(float(t_sec) / float(time_scale)))


def raw_to_sec(t_raw: int, time_scale: float) -> float:
    return float(t_raw) * float(time_scale)


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
# GPU polarity frame (2ch)
# ---------------------------
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

def normalize_frame(frame_2hw: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return frame_2hw
    if mode == "max":
        m = torch.amax(frame_2hw)
        return frame_2hw / torch.clamp(m, min=1.0)
    if mode == "logmax":
        f = torch.log1p(frame_2hw)
        m = torch.amax(f)
        return f / torch.clamp(m, min=1e-6)
    raise ValueError(mode)


# ---------------------------
# GPU MCTS
# ---------------------------
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
    def __init__(self, ref_dir: Path, pattern: str, num_refs: int, cache_size: int, max_kpts: int):
        self.ref_dir = Path(ref_dir)
        self.pattern = pattern
        self.max_kpts = max_kpts
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.num_refs = num_refs

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

import matplotlib.pyplot as plt
import cv2

def debug_plot_keypoints(mcts_img, kpts_yx, save_path="debug_kpts.png"):
    """
    mcts_img: The tensor or numpy array [H, W] or [H, W, 3]
    kpts_yx: The keypoints in [N, 2] format (y, x)
    """
    # print(mcts_img.shape, kpts_yx.shape)
    if torch.is_tensor(mcts_img):
        img = mcts_img.detach().cpu().numpy()
    else:
        img = mcts_img
    
    # print(img.shape)

    # Normalize image for visualization if it's raw event data
    img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for y, x in kpts_yx:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

    cv2.imwrite(save_path, img)
    print(f"[DEBUG] Keypoint visualization saved to {save_path}")

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
# Main
# ---------------------------
def main():
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--hdf5", type=str, required=True)
    # ap.add_argument("--backbone-ckpt", type=str, required=True)
    # ap.add_argument("--norm", type=str, default="logmax", choices=["none", "max", "logmax"])
    # ap.add_argument("--resize", type=str, default="nearest", choices=["nearest", "bilinear"])
    # ap.add_argument("--amp", action="store_true")

    # ap.add_argument("--ref-feats", type=str, required=True)
    # ap.add_argument("--retrieval-k", type=int, default=33)

    # ap.add_argument("--se-config", type=str, required=True)
    # ap.add_argument("--se-weights", type=str, required=True)
    # ap.add_argument("--se-topk", type=int, default=170)

    # ap.add_argument("--do-rerank", action="store_true")
    # ap.add_argument("--ref-kp-dir", type=str, default=None)
    # ap.add_argument("--ref-kp-pattern", type=str, default="mcts_{:05d}.feat.npz")
    # ap.add_argument("--ref-kp-preload", action="store_true")
    # ap.add_argument("--ref-kp-cache", type=int, default=2048)
    # ap.add_argument("--ransac-thresh", type=float, default=5.0)
    # ap.add_argument("--inlier-weight", type=float, default=0.05)
    # ap.add_argument("--match-ratio", type=float, default=0.8)

    # ap.add_argument("--mcts-windows-ms", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    # ap.add_argument("--dt-ms", type=float, default=50.0)
    # ap.add_argument("--target-hz", type=float, default=20.0)
    # ap.add_argument("--realtime", action="store_true")
    # ap.add_argument("--chunk-size", type=int, default=250_000)
    # ap.add_argument("--time-scale", type=float, default=1e-9)
    # ap.add_argument("--start-time", type=float, default=None)
    # ap.add_argument("--max-frames", type=int, default=None)

    # ap.add_argument("--height", type=int, default=0)
    # ap.add_argument("--width", type=int, default=0)
    # ap.add_argument("--infer-full-scan", action="store_true")

    # ap.add_argument("--warmup", type=int, default=50)
    # ap.add_argument("--gt-file", type=str, default=None,
    #                 help="Path to the ground truth file")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hdf5",
        type=str,
        default="/media/adam/vprdatasets/eventlab/brisbane_event/sunset1/sunset1.hdf5",
    )
    ap.add_argument(
        "--backbone-ckpt",
        type=str,
        default="./eventgem/ckpt/pr.pt",
    )
    ap.add_argument("--norm", type=str, default="logmax", choices=["none", "max", "logmax"])
    ap.add_argument("--resize", type=str, default="nearest", choices=["nearest", "bilinear"])
    ap.add_argument("--amp", action="store_true")

    ap.add_argument(
        "--ref-feats",
        type=str,
        default="eventgem/features/brisbane_event/sunset2-sunset1/brisbane_event_sunset2_features.pt",
    )
    ap.add_argument("--retrieval-k", type=int, default=50)

    ap.add_argument(
        "--se-config",
        type=str,
        default="./eventgem/external/superevent/config/super_event.yaml",
    )
    ap.add_argument(
        "--se-weights",
        type=str,
        default="./eventgem/external/superevent/saved_models/super_event_weights.pth",
    )
    ap.add_argument("--se-topk", type=int, default=170)

    # default rerank ON, with an escape hatch flag to disable it
    ap.add_argument("--do-rerank", dest="do_rerank", action="store_true", default=True)
    ap.add_argument("--no-rerank", dest="do_rerank", action="store_false")

    ap.add_argument(
        "--ref-kp-dir",
        type=str,
        default="./eventgem/keypoints/brisbane_event/kps_sunset2",
    )
    ap.add_argument("--ref-kp-pattern", type=str, default="mcts_{:05d}.feat.npz")
    ap.add_argument("--ref-kp-preload", action="store_true")
    ap.add_argument("--ref-kp-cache", type=int, default=2048)
    ap.add_argument("--ransac-thresh", type=float, default=5.0)
    ap.add_argument("--inlier-weight", type=float, default=0.05)
    ap.add_argument("--match-ratio", type=float, default=0.8)

    ap.add_argument("--mcts-windows-ms", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    ap.add_argument("--dt-ms", type=float, default=50.0)
    ap.add_argument("--target-hz", type=float, default=20.0)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--time-scale", type=float, default=1e-9)
    ap.add_argument("--start-time", type=float, default=1587452582.35)
    ap.add_argument("--max-frames", type=int, default=None)

    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--infer-full-scan", action="store_true")

    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument(
        "--gt-file",
        type=str,
        default="/media/adam/vprdatasets/eventgem/ground_truth/sunset2_sunset1_GT.npy",
        help="Path to the ground truth file",
    )

    args = ap.parse_args()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    with h5py.File(hdf5_path, "r") as f:
        x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)
        if args.height > 0 and args.width > 0:
            H, W = args.height, args.width
        else:
            H, W = infer_resolution(x_dset, y_dset, args.chunk_size, args.infer_full_scan)
        
        t0_raw = int(t_dset[0]); tN_raw = int(t_dset[-1])
        print(f"[INFO] Sensor: {H}x{W}")

    vit = load_vit_backbone(args.backbone_ckpt, device)
    vitH, vitW = infer_vit_input_hw(vit)
    print(f"[INFO] ViT expects ~ {vitH}x{vitW}")

    se_model, se_cfg = build_superevent_model(Path(args.se_config), Path(args.se_weights), device)
    from models.util import fast_nms

    off_top, off_left, off_bottom, off_right, h_end, w_end, Hc, Wc = compute_superevent_crop_offsets(H, W, se_cfg)
    print(f"[INFO] SuperEvent crop: top={off_top} left={off_left} -> {Hc}x{Wc}")

    ref_db = load_ref_vit_db(args.ref_feats, device=device, dtype=torch.float16 if args.amp else torch.float32)
    num_refs = int(ref_db.shape[0])
    print(f"[INFO] Loaded ref DB: {num_refs} feats on GPU")

    ref_store = None
    if args.do_rerank:
        if args.ref_kp_dir is None:
            raise ValueError("--do-rerank requires --ref-kp-dir")
        
        ref_store = BatchedRefStore(
            ref_dir=Path(args.ref_kp_dir),
            pattern=args.ref_kp_pattern,
            num_refs=num_refs,
            cache_size=args.ref_kp_cache,
            max_kpts=args.se_topk
        )
        print(f"[INFO] Ref kp store: {args.ref_kp_dir} (CPU Cache -> Batched GPU)")

    windows_sec = torch.tensor(np.array(args.mcts_windows_ms, dtype=np.float32) * 1e-3, device=device)

    stream_vit = torch.cuda.Stream()
    stream_se = torch.cuda.Stream()
    join_stream = torch.cuda.current_stream()
    
    vit0 = torch.cuda.Event(True); vit1 = torch.cuda.Event(True)
    se0 = torch.cuda.Event(True);  se1 = torch.cuda.Event(True)
    j0 = torch.cuda.Event(True);   j1 = torch.cuda.Event(True)
    ret0 = torch.cuda.Event(True); ret1 = torch.cuda.Event(True)

    t_read_list, t_vit_list, t_se_list, t_rerank_list, t_total_list = [], [], [], [], []
    n_events_list = []

    target_period = 1.0 / max(1e-6, args.target_hz)
    wall0 = time.perf_counter()

    print("[INFO] Starting Loop...", flush=True)

    reranked_cols = []
    sims = []
    with torch.inference_mode():
        for (w0_sec, w1_sec, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms) in stream_event_windows_raw(
            hdf5_path, args.dt_ms, args.chunk_size, args.time_scale, args.start_time, args.max_frames
        ):
            cpu0 = time.perf_counter()
            n_events = int(x.size)
            j0.record(join_stream)
            
            if x.size == 0:
                continue

            with torch.cuda.stream(stream_vit):
                vit0.record(stream_vit)
                pol_2hw, _ = gpu_polarity_frame_2ch(x, y, p, H, W, device)
                # pol_2hw = normalize_frame(pol_2hw, args.norm)

                inp = vit_preprocess_like_dataloader(pol_2hw, out_hw=(vitH, vitW))
                if (H != vitH) or (W != vitW):
                    mode = "nearest" if args.resize == "nearest" else "bilinear"
                    inp = F.interpolate(inp, size=(vitH, vitW), mode=mode, align_corners=False if mode=="bilinear" else None)

                if args.amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        q_desc_vit = vit_gem_descriptor(vit, inp)
                else:
                    q_desc_vit = vit_gem_descriptor(vit, inp)

                ret0.record(stream_vit)
                top_idx_t, top_dist_t, sims_t = retrieve_topk(ref_db, q_desc_vit, k=int(args.retrieval_k), return_sims=True)
                ret1.record(stream_vit)
                vit1.record(stream_vit)
                sims.append(sims_t.cpu())

            with torch.cuda.stream(stream_se):
                se0.record(stream_se)
                mcts = gpu_mcts(x, y, t_raw, p, H, W, int(t_ref_raw), float(args.time_scale), windows_sec, device)
                mcts = mcts[:, off_top:h_end, off_left:w_end] 
                
                pred = se_model(mcts.unsqueeze(0))
                prob, desc_map = pred['prob'], pred['descriptors']
                
                kpts_all, _ = fast_nms(prob, se_cfg, top_k=int(args.se_topk))
                kpts_yx = kpts_all[0]

                # debug_plot_keypoints(mcts[0], kpts_yx.cpu().numpy(), save_path=f"debug_kpts_{frame_idx}.png")

                q_k_desc = sample_descriptors_at_kpts(kpts_yx.float(), desc_map)
                se1.record(stream_se)

            join_stream.wait_stream(stream_vit)
            join_stream.wait_stream(stream_se)
            j1.record(join_stream)
            torch.cuda.synchronize()
            
            vit_ms = vit0.elapsed_time(vit1)
            se_ms = se0.elapsed_time(se1)

            t_rerank0 = time.perf_counter()
            best_idx = int(top_idx_t[0].item()) if top_idx_t.numel() else -1
            best_inl = 0
            
            if args.do_rerank and top_idx_t.numel() > 0 and kpts_yx.numel() > 10:
                q_xy = kpts_yx[:, [1,0]].float()
                q_xy[:,0] += float(off_left)
                q_xy[:,1] += float(off_top)
                
                cand_ids = top_idx_t.cpu().numpy().astype(np.int64)
                cand_dist_val = top_dist_t.cpu().numpy()
                
                inlier_counts = batched_ransac_rerank(
                    q_xy, q_k_desc, ref_store, cand_ids, 
                    max_matches=170, ratio_thresh=float(args.match_ratio)
                )

                # after you compute inlier_counts for cand_ids, build the new column:
                new_col = build_reranked_column_from_sims(
                    sims_t=sims_t,
                    cand_ids=cand_ids,
                    inlier_counts=inlier_counts,
                    inlier_weight=float(args.inlier_weight),
                )
                # then stash it for later matrix assembly (or write to disk)
                reranked_cols.append(new_col)

                final_scores = cand_dist_val - (inlier_counts * args.inlier_weight)
                best_arg = np.argmin(final_scores)
                best_idx = cand_ids[best_arg]
                best_inl = inlier_counts[best_arg]

            t_rerank = (time.perf_counter() - t_rerank0) * 1000.0
            t_total = (time.perf_counter() - cpu0) * 1000.0

            if args.realtime:
                elapsed = time.perf_counter() - cpu0
                time.sleep(max(0.0, target_period - elapsed))

            if frame_idx >= args.warmup:
                t_read_list.append(t_read_ms)
                t_vit_list.append(vit_ms)
                t_se_list.append(se_ms)
                t_rerank_list.append(t_rerank)
                t_total_list.append(t_total)
                n_events_list.append(n_events)

            if (frame_idx % 100) == 0:
                hz = 1000.0 / max(1e-6, t_total)
                print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} vit={vit_ms:.1f} se={se_ms:.1f} rerank={t_rerank:.1f} total={t_total:.1f}ms ({hz:.1f} Hz) best={best_idx} inl={best_inl}", flush=True)

    wall_s = time.perf_counter() - wall0
    n_frames = len(t_total_list)

    print("\n========== SUMMARY ==========")
    print(f"Frames: {n_frames} | Wall: {wall_s:.2f}s | Avg FPS: {n_frames/wall_s:.2f}")
    t_total_np = np.array(t_total_list)
    print(summarize_ms(np.array(t_read_list), "Read"))
    print(summarize_ms(np.array(t_vit_list), "ViT (GPU)"))
    print(summarize_ms(np.array(t_se_list), "SE (GPU)"))
    print(summarize_ms(np.array(t_rerank_list), "Rerank (Batch GPU)"))
    print(summarize_ms(t_total_np, "Total End2End"))
    print(f"Over budget ({args.dt_ms}ms): {np.sum(t_total_np > args.dt_ms)}/{n_frames}")

    S_vit = np.stack(sims, axis=0).squeeze()  # [N_ref] or [N_ref, N_queries]
    S_vit = S_vit.T  # [N_queries, N_ref]
    np.save("S_vit_nonorm.npy", S_vit)
    # stack the reranked cols
    if args.gt_file is not None:
        reranked_cols_stack = np.stack(reranked_cols, axis=0)
        reranked_cols_stack = reranked_cols_stack.T
        S_in = 1-reranked_cols_stack
        print(S_in.shape)
        np.save("S_in.npy", S_in)
        K = [1, 5, 10]
        # load ground truth file
        gt = np.load(args.gt_file)
        # resize to match shape of reranked_cols_stack if needed
        from skimage.transform import resize
        gt_resized = resize(gt, S_in.shape, order=0, preserve_range=True, anti_aliasing=False)
        from prettytable import PrettyTable
        table = PrettyTable()

        # add columns for each K
        for k in K:
            table.add_column(f"Recall@{k}", [recallAtK(S_in, gt_resized, K=k)])

        print(table)

if __name__ == "__main__":
    main()