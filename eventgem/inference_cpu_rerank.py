#!/usr/bin/env python3
import argparse
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import yaml
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import sys
import cv2
from collections import OrderedDict

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
# Streaming windows (raw units)
# ---------------------------
def stream_event_windows_raw(
    hdf5_path: Path,
    dt_ms: float,
    chunk_size: int,
    time_scale: float,
    start_time_sec: Optional[float],
    max_frames: Optional[int],
):
    """
    Yields:
      (w0_sec, w1_sec, t_ref_raw, x_win, y_win, t_win_raw, p_win, frame_idx, t_read_ms)
    """
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

        while read_idx < N and (t_buf.size == 0 or t_buf_max < w_end_raw):
            end_idx = min(N, read_idx + chunk_size)

            x_chunk = x_dset[read_idx:end_idx].astype(np.int64, copy=False)
            y_chunk = y_dset[read_idx:end_idx].astype(np.int64, copy=False)
            t_chunk = t_dset[read_idx:end_idx].astype(np.int64, copy=False)
            p_chunk = p_dset[read_idx:end_idx].astype(np.int8, copy=False)

            read_idx = end_idx
            if t_chunk.size == 0:
                continue

            if x_buf.size == 0:
                x_buf, y_buf, t_buf, p_buf = x_chunk, y_chunk, t_chunk, p_chunk
            else:
                x_buf = np.concatenate((x_buf, x_chunk))
                y_buf = np.concatenate((y_buf, y_chunk))
                t_buf = np.concatenate((t_buf, t_chunk))
                p_buf = np.concatenate((p_buf, p_chunk))

            t_buf_max = int(max(t_buf_max, int(np.max(t_chunk))))

        if t_buf.size:
            keep = t_buf >= w_start_raw
            x_buf, y_buf, t_buf, p_buf = x_buf[keep], y_buf[keep], t_buf[keep], p_buf[keep]
            t_buf_max = int(np.max(t_buf)) if t_buf.size else -1

        if t_buf.size:
            in_win = (t_buf >= w_start_raw) & (t_buf < w_end_raw)
            x_win = x_buf[in_win]
            y_win = y_buf[in_win]
            t_win_raw = t_buf[in_win]
            p_win = p_buf[in_win]

            leftover = t_buf >= w_end_raw
            x_buf, y_buf, t_buf, p_buf = x_buf[leftover], y_buf[leftover], t_buf[leftover], p_buf[leftover]
            t_buf_max = int(np.max(t_buf)) if t_buf.size else -1
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
            w_end_raw,  # reference time (raw)
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
    # uses BACKBONE_ROOT on sys.path
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
    if "patch_embed.proj.weight" in getattr(msg, "missing_keys", []):
        raise RuntimeError("Backbone load failed: patch_embed.proj.weight missing.")

    backbone.to(device).eval()
    return backbone


def infer_vit_input_hw(backbone) -> Tuple[int, int]:
    N = int(backbone.pos_embed.shape[1])
    g = int(round(math.sqrt(N)))
    if g * g != N:
        raise RuntimeError(f"pos_embed length {N} not square; cannot infer input size.")
    patch = getattr(backbone.patch_embed, "patch_size", 16)
    patch = int(patch[0]) if isinstance(patch, (tuple, list)) else int(patch)
    img = g * patch
    return img, img


@torch.no_grad()
def vit_gem_descriptor(backbone, x_bchw: torch.Tensor) -> torch.Tensor:
    t = backbone.patch_embed(x_bchw)  # [B,N,C]
    t = t + backbone.pos_embed
    t = torch.cat((backbone.tokens.expand(t.shape[0], -1, -1), t), dim=1)

    for blk in backbone.blocks:
        t = blk(t)
    t = backbone.norm(t)

    patch_tokens = t[:, 2:, :]  # [B,N,C]
    B, N, C = patch_tokens.shape
    g = int(round(math.sqrt(N)))
    patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, g, g)

    p = 5.0
    gem = F.avg_pool2d((patch_tokens.clamp(min=1e-6)).pow(p), (g, g)).pow(1.0 / p)
    return gem.squeeze(-1).squeeze(-1)  # [B,C]


# ---------------------------
# GPU polarity frame (2ch) via bincount
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
    p = torch.from_numpy(p_np).to(device=device, dtype=torch.int8)

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not torch.any(valid):
        return torch.zeros((2, H, W), device=device, dtype=torch.float32), 0

    x = x[valid]; y = y[valid]; p = p[valid]
    n_valid = int(x.numel())

    flat = H * W
    lin = y * W + x
    ch = (p > 0).to(torch.int64)  # 0=neg, 1=pos
    idx = lin + ch * flat

    counts = torch.bincount(idx, minlength=2 * flat).to(torch.float32)
    frame = counts.view(2, flat).view(2, H, W)
    return frame, n_valid


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
# GPU MCTS (2*len(windows) channels)
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

    # t_rel <= 0
    t_rel = (t - int(t_ref_raw)).to(torch.float32) * float(time_scale)

    # group: 0=POS, 1=NEG
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
# SuperEvent crop logic (EXACT from EventGeMMCTS)
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
            raise FileNotFoundError(
                f"SuperEvent backbone='{backbone_name}' but could not find backbone yaml. "
                f"Tried: {[str(p) for p in candidates]}"
            )
        with open(bb_path, "r") as f:
            bb_cfg = yaml.safe_load(f)

        cfg.update(bb_cfg)
        if "backbone_config" in cfg and "input_channels" in cfg:
            cfg["backbone_config"]["input_channels"] = cfg["input_channels"]

    if "backbone_config" not in cfg:
        raise KeyError(f"`backbone_config` missing. Loaded keys: {list(cfg.keys())}")

    return cfg


def build_superevent_model(config_path: Path, weights_path: Path, device: torch.device):
    # relies on SUPEREVENT_ROOT on sys.path so "models" resolves
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
    k_xy = keypoints_yx[:, [1, 0]]  # (x,y)

    grid = torch.empty((1, 1, k_xy.shape[0], 2), device=descriptors.device, dtype=descriptors.dtype)
    grid[0, 0, :, 0] = 2.0 * k_xy[:, 0] / (Wc - 1) - 1.0
    grid[0, 0, :, 1] = 2.0 * k_xy[:, 1] / (Hc - 1) - 1.0

    samp = F.grid_sample(descriptors, grid, mode="bilinear", align_corners=True)  # (1,D,1,N)
    samp = samp[0, :, 0, :].t()  # (N,D)
    samp = F.normalize(samp, p=2, dim=1)
    return samp


# ---------------------------
# Retrieval DB (ViT)
# ---------------------------
def load_ref_vit_db(ref_feats_path: str, device: torch.device, dtype: torch.dtype):
    ref = torch.load(ref_feats_path, map_location="cpu")
    if not isinstance(ref, torch.Tensor):
        ref = torch.tensor(ref, dtype=torch.float32)

    if ref.ndim != 2:
        raise ValueError(f"ref feats must be [R,C], got {tuple(ref.shape)}")

    ref = ref.to(device=device, dtype=dtype)
    ref = F.normalize(ref, p=2, dim=1)
    return ref  # [R,C]


@torch.no_grad()
def retrieve_topk(ref_db: torch.Tensor, q_desc: torch.Tensor, k: int):
    """
    ref_db: [R,C] L2-normalized (on GPU)
    q_desc: [1,C] or [C] (on GPU)
    returns: top_idx [k] (GPU int64), top_dist [k] (GPU float32) where dist = 1 - cos
    """
    if q_desc.ndim == 2:
        q = q_desc[0]
    else:
        q = q_desc
    q = q.to(dtype=ref_db.dtype)
    q = F.normalize(q, p=2, dim=0)

    sims = torch.matmul(ref_db, q)  # [R]
    kk = min(int(k), int(sims.numel()))
    top_sim, top_idx = torch.topk(sims, k=kk, largest=True)
    top_dist = (1.0 - top_sim).to(torch.float32)
    return top_idx, top_dist


# ---------------------------
# Rerank (CPU/OpenCV) against precomputed ref .feat.npz
# ---------------------------
def load_event_features_npz(root_dir: Path, idx: int, pattern: str):
    path = root_dir / pattern.format(idx)
    if not path.exists():
        return None
    try:
        data = np.load(str(path))
        if "keypoints" in data:
            kpts = data["keypoints"].astype(np.float32)  # (N,2) XY
        else:
            return None

        if "descriptors" in data:
            desc = data["descriptors"].astype(np.float32)
        else:
            return None

        if kpts.shape[0] == 0 or desc.shape[0] == 0:
            return None
        if desc.ndim == 1:
            desc = desc.reshape(1, -1)

        return {"kpts": kpts, "desc": desc}
    except Exception:
        return None


class RefKpStore:
    """
    Either preload everything (if feasible), or use LRU cache.
    """
    def __init__(self, ref_dir: Path, pattern: str, num_refs: int, preload: bool, cache_size: int):
        self.ref_dir = Path(ref_dir)
        self.pattern = pattern
        self.num_refs = int(num_refs)
        self.preload = bool(preload)
        self.cache_size = int(cache_size)

        self._all = None
        self._lru = OrderedDict()

        if self.preload:
            self._all = [None] * self.num_refs
            for i in range(self.num_refs):
                self._all[i] = load_event_features_npz(self.ref_dir, i, self.pattern)

    def get(self, idx: int):
        idx = int(idx)
        if self._all is not None:
            return self._all[idx]

        if idx in self._lru:
            self._lru.move_to_end(idx)
            return self._lru[idx]

        v = load_event_features_npz(self.ref_dir, idx, self.pattern)
        self._lru[idx] = v
        self._lru.move_to_end(idx)
        if len(self._lru) > self.cache_size:
            self._lru.popitem(last=False)
        return v


def compute_inliers_2d(q_kpts_xy, q_desc, r_kpts_xy, r_desc, matcher, ransac_thresh):
    if q_kpts_xy is None or q_desc is None or r_kpts_xy is None or r_desc is None:
        return 0
    if len(q_kpts_xy) < 4 or len(r_kpts_xy) < 4:
        return 0

    try:
        matches = matcher.knnMatch(q_desc, r_desc, k=2)
    except cv2.error:
        return 0

    good = []
    for m_n in matches:
        if len(m_n) == 2 and m_n[0].distance < 0.8 * m_n[1].distance:
            good.append(m_n[0])

    if len(good) < 4:
        return 0

    src_pts = np.float32([q_kpts_xy[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([r_kpts_xy[m.trainIdx] for m in good]).reshape(-1, 1, 2)

    try:
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    except cv2.error:
        return 0

    if mask is None:
        return 0
    return int(np.sum(mask))


def rerank_topk_inliers(
    cand_idx: np.ndarray,
    cand_dist: np.ndarray,
    q_kpts_xy: np.ndarray,
    q_desc: np.ndarray,
    ref_store: RefKpStore,
    ransac_thresh: float,
    inlier_weight: float,
):
    """
    Rerank ONLY within candidates.
    Returns: (idx_sorted, dist_sorted, inliers_sorted)
    """
    order0 = np.argsort(cand_dist)
    cand_idx = cand_idx[order0]
    cand_dist = cand_dist[order0]

    if q_kpts_xy is None or q_desc is None or q_kpts_xy.shape[0] < 4:
        return cand_idx, cand_dist, np.zeros_like(cand_dist, dtype=np.int32)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    new_dist = cand_dist.copy()
    inliers = np.zeros_like(cand_dist, dtype=np.int32)

    for j, r_idx in enumerate(cand_idx):
        r = ref_store.get(int(r_idx))
        if r is None:
            continue

        num_in = compute_inliers_2d(
            q_kpts_xy, q_desc,
            r["kpts"], r["desc"],
            matcher, ransac_thresh
        )
        inliers[j] = num_in
        if num_in > 0:
            new_dist[j] = cand_dist[j] - (num_in * inlier_weight)

    order = np.argsort(new_dist)
    return cand_idx[order], new_dist[order], inliers[order]


# ---------------------------
# Stats formatting
# ---------------------------
def summarize_ms(arr: np.ndarray, name: str) -> str:
    if arr.size == 0:
        return f"{name}: (no data)"
    return f"{name} ms | mean {arr.mean():.2f}  med {np.median(arr):.2f}  p95 {np.percentile(arr,95):.2f}  max {arr.max():.2f}"


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=str, required=True)

    # ViT
    ap.add_argument("--backbone-ckpt", type=str, required=True)
    ap.add_argument("--norm", type=str, default="logmax", choices=["none", "max", "logmax"])
    ap.add_argument("--resize", type=str, default="nearest", choices=["nearest", "bilinear"])
    ap.add_argument("--amp", action="store_true")

    # Retrieval DB
    ap.add_argument("--ref-feats", type=str, required=True, help="Reference ViT features .pt (Tensor [R,512])")
    ap.add_argument("--retrieval-k", type=int, default=33)

    # SuperEvent
    ap.add_argument("--se-config", type=str, required=True)
    ap.add_argument("--se-weights", type=str, required=True)
    ap.add_argument("--se-topk", type=int, default=1024)

    # Rerank (optional)
    ap.add_argument("--do-rerank", action="store_true")
    ap.add_argument("--ref-kp-dir", type=str, default=None)
    ap.add_argument("--ref-kp-pattern", type=str, default="mcts_{:05d}.feat.npz")
    ap.add_argument("--ref-kp-preload", action="store_true")
    ap.add_argument("--ref-kp-cache", type=int, default=20000)
    ap.add_argument("--ransac-thresh", type=float, default=5.0)
    ap.add_argument("--inlier-weight", type=float, default=0.05)

    # MCTS windows
    ap.add_argument("--mcts-windows-ms", type=int, nargs="+", default=[10, 20, 30, 40, 50])

    # Stream/windowing
    ap.add_argument("--dt-ms", type=float, default=50.0)
    ap.add_argument("--target-hz", type=float, default=20.0)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--time-scale", type=float, default=1e-9)
    ap.add_argument("--start-time", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)

    # Sensor size
    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--infer-full-scan", action="store_true")

    ap.add_argument("--warmup", type=int, default=50)

    args = ap.parse_args()

    hdf5_path = Path(args.hdf5)
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for this benchmark.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Dataset H/W
    with h5py.File(hdf5_path, "r") as f:
        x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)

        if args.height > 0 and args.width > 0:
            H, W = args.height, args.width
        else:
            H, W = infer_resolution(x_dset, y_dset, args.chunk_size, args.infer_full_scan)

        t0_raw = int(t_dset[0]); tN_raw = int(t_dset[-1])
        print("[INFO] datasets:",
              "x", x_dset.name, "y", y_dset.name, "t", t_dset.name, "p", p_dset.name)
        print(f"[INFO] sensor HxW = {H}x{W}")
        print(f"[INFO] raw t range: [{t0_raw}, {tN_raw}] sec: [{raw_to_sec(t0_raw,args.time_scale):.6f}, {raw_to_sec(tN_raw,args.time_scale):.6f}]")

    # Models
    vit = load_vit_backbone(args.backbone_ckpt, device)
    vitH, vitW = infer_vit_input_hw(vit)
    print(f"[INFO] ViT expects ~ {vitH}x{vitW}")

    se_model, se_cfg = build_superevent_model(Path(args.se_config), Path(args.se_weights), device)
    from models.util import fast_nms  # import after sys.path is set up

    # SuperEvent crop offsets (exact EventGeMMCTS logic)
    off_top, off_left, off_bottom, off_right, h_end, w_end, Hc, Wc = compute_superevent_crop_offsets(H, W, se_cfg)
    print(f"[INFO] SuperEvent crop: top={off_top} left={off_left} bottom={off_bottom} right={off_right} -> {Hc}x{Wc}")

    # Ref DB (GPU)
    ref_db = load_ref_vit_db(args.ref_feats, device=device, dtype=torch.float16)
    num_refs = int(ref_db.shape[0])
    print(f"[INFO] Loaded ref DB: {num_refs} feats on GPU from {args.ref_feats}")

    # Ref keypoint store (CPU)
    ref_store = None
    if args.do_rerank:
        if args.ref_kp_dir is None:
            raise ValueError("--do-rerank requires --ref-kp-dir")
        ref_store = RefKpStore(
            ref_dir=Path(args.ref_kp_dir),
            pattern=args.ref_kp_pattern,
            num_refs=num_refs,
            preload=args.ref_kp_preload,
            cache_size=args.ref_kp_cache,
        )
        print(f"[INFO] Ref kp store: dir={args.ref_kp_dir} preload={args.ref_kp_preload} cache={args.ref_kp_cache}")

    windows_sec = torch.tensor(np.array(args.mcts_windows_ms, dtype=np.float32) * 1e-3, device=device)

    # Streams
    stream_vit = torch.cuda.Stream()
    stream_se = torch.cuda.Stream()
    join_stream = torch.cuda.current_stream()

    # CUDA events (timing)
    vit0 = torch.cuda.Event(True); vit1 = torch.cuda.Event(True)
    se0 = torch.cuda.Event(True);  se1 = torch.cuda.Event(True)
    j0 = torch.cuda.Event(True);   j1 = torch.cuda.Event(True)

    # Extra: retrieval timing on stream A
    ret0 = torch.cuda.Event(True); ret1 = torch.cuda.Event(True)

    # Stats
    t_read_list = []
    t_vit_list = []
    t_ret_list = []
    t_se_list = []
    t_join_list = []
    t_sync_list = []     # CPU time up to GPU sync
    t_rerank_list = []   # CPU rerank time
    t_total_list = []    # CPU total time (sync + rerank)
    n_events_list = []
    n_valid_list = []
    n_kpts_list = []

    target_period = 1.0 / max(1e-6, args.target_hz)

    wall0 = time.perf_counter()

    with torch.inference_mode():
        for (w0_sec, w1_sec, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms) in stream_event_windows_raw(
            hdf5_path,
            dt_ms=args.dt_ms,
            chunk_size=args.chunk_size,
            time_scale=args.time_scale,
            start_time_sec=args.start_time,
            max_frames=args.max_frames,
        ):
            cpu0 = time.perf_counter()
            n_events = int(x.size)

            # joined timing starts on default stream
            j0.record(join_stream)

            # --------------------
            # Stream A: polarity -> ViT -> GeM -> retrieval (GPU)
            # --------------------
            with torch.cuda.stream(stream_vit):
                vit0.record(stream_vit)

                pol_2hw, n_valid = gpu_polarity_frame_2ch(x, y, p, H, W, device)
                pol_2hw = normalize_frame(pol_2hw, args.norm)

                inp = pol_2hw.unsqueeze(0)  # [1,2,H,W]
                if (H != vitH) or (W != vitW):
                    mode = "nearest" if args.resize == "nearest" else "bilinear"
                    inp = F.interpolate(
                        inp, size=(vitH, vitW), mode=mode,
                        align_corners=False if mode == "bilinear" else None
                    )

                if args.amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        q_desc = vit_gem_descriptor(vit, inp)  # [1,C]
                else:
                    q_desc = vit_gem_descriptor(vit, inp)

                # retrieval timing (subset of vit path)
                ret0.record(stream_vit)
                top_idx_t, top_dist_t = retrieve_topk(ref_db, q_desc, k=int(args.retrieval_k))
                ret1.record(stream_vit)

                vit1.record(stream_vit)

            # --------------------
            # Stream B: MCTS -> SuperEvent -> NMS -> sample descs (GPU)
            # --------------------
            with torch.cuda.stream(stream_se):
                se0.record(stream_se)

                mcts = gpu_mcts(x, y, t_raw, p, H, W, int(t_ref_raw), float(args.time_scale), windows_sec, device)
                mcts = mcts[:, off_top:h_end, off_left:w_end]  # (C,Hc,Wc)
                batch = mcts.unsqueeze(0)

                pred = se_model(batch)
                if isinstance(pred, tuple):
                    pred = {"prob": pred[0], "descriptors": pred[1]}

                prob = pred["prob"]                 # (1,1,Hc,Wc)
                desc_map = pred["descriptors"]      # (1,D,Hc,Wc)

                kpts_all, scores_all = fast_nms(prob, se_cfg, top_k=int(args.se_topk))
                kpts = kpts_all[0]                  # (N,2) (y,x) on cropped coords

                q_k_desc = sample_descriptors_at_kpts(kpts.float(), desc_map)  # (N,D)

                se1.record(stream_se)

            # join: wait for both streams, then stop joined timer
            join_stream.wait_stream(stream_vit)
            join_stream.wait_stream(stream_se)
            j1.record(join_stream)

            # sync point (keeps timings comparable / safe for CPU rerank)
            torch.cuda.synchronize()

            cpu_sync = time.perf_counter()

            # GPU timings
            vit_ms = vit0.elapsed_time(vit1)
            ret_ms = ret0.elapsed_time(ret1)
            se_ms = se0.elapsed_time(se1)
            join_ms = j0.elapsed_time(j1)

            # CPU sync time (up to GPU completion)
            sync_ms = (cpu_sync - cpu0) * 1000.0

            # Pull candidates to CPU
            cand_idx = top_idx_t.detach().cpu().numpy().astype(np.int32)
            cand_dist = top_dist_t.detach().cpu().numpy().astype(np.float32)

            # Build query keypoints+descriptors for CPU rerank:
            # kpts are (y,x) cropped -> convert to full-frame XY
            if kpts.numel() == 0:
                q_kpts_xy = None
                q_desc_np = None
            else:
                k = kpts.detach().cpu().numpy().astype(np.float32)  # (N,2) yx cropped
                k[:, 0] += off_top
                k[:, 1] += off_left
                q_kpts_xy = np.stack([k[:, 1], k[:, 0]], axis=1).astype(np.float32)  # (N,2) XY
                q_desc_np = q_k_desc.detach().cpu().numpy().astype(np.float32)

            # Optional rerank on CPU
            rerank_ms = 0.0
            best_idx = int(cand_idx[0]) if cand_idx.size else -1
            best_dist = float(cand_dist[0]) if cand_dist.size else float("inf")
            best_inliers = 0

            if args.do_rerank and (cand_idx.size > 0):
                t0 = time.perf_counter()
                cand_idx_r, cand_dist_r, inliers_r = rerank_topk_inliers(
                    cand_idx=cand_idx,
                    cand_dist=cand_dist,
                    q_kpts_xy=q_kpts_xy,
                    q_desc=q_desc_np,
                    ref_store=ref_store,
                    ransac_thresh=float(args.ransac_thresh),
                    inlier_weight=float(args.inlier_weight),
                )
                rerank_ms = (time.perf_counter() - t0) * 1000.0

                best_idx = int(cand_idx_r[0])
                best_dist = float(cand_dist_r[0])
                best_inliers = int(inliers_r[0])

            cpu_end = time.perf_counter()
            total_ms = (cpu_end - cpu0) * 1000.0

            # optional realtime pacing
            if args.realtime:
                elapsed = cpu_end - cpu0
                time.sleep(max(0.0, target_period - elapsed))

            # kpt count
            n_kpts = int(kpts.shape[0])

            if frame_idx >= args.warmup:
                t_read_list.append(t_read_ms)
                t_vit_list.append(vit_ms)
                t_ret_list.append(ret_ms)
                t_se_list.append(se_ms)
                t_join_list.append(join_ms)
                t_sync_list.append(sync_ms)
                t_rerank_list.append(rerank_ms)
                t_total_list.append(total_ms)
                n_events_list.append(n_events)
                n_valid_list.append(int(n_valid))
                n_kpts_list.append(n_kpts)

            if (frame_idx % 100) == 0:
                hz_inst = 1000.0 / max(1e-6, total_ms)
                print(
                    f"[LIVE] frame={frame_idx:6d} ev={n_events:7d} valid={int(n_valid):7d} kpts={n_kpts:5d} "
                    f"read={t_read_ms:6.2f}ms vit={vit_ms:6.2f}ms ret={ret_ms:6.2f}ms "
                    f"se={se_ms:6.2f}ms join_gpu={join_ms:6.2f}ms "
                    f"sync={sync_ms:6.2f}ms rerank={rerank_ms:6.2f}ms total={total_ms:6.2f}ms "
                    f"best={best_idx:6d} d={best_dist:7.4f} inl={best_inliers:4d} (~{hz_inst:5.1f} Hz)"
                )

    wall1 = time.perf_counter()
    wall_s = wall1 - wall0

    t_read = np.array(t_read_list, dtype=np.float64)
    t_vit = np.array(t_vit_list, dtype=np.float64)
    t_ret = np.array(t_ret_list, dtype=np.float64)
    t_se = np.array(t_se_list, dtype=np.float64)
    t_join = np.array(t_join_list, dtype=np.float64)
    t_sync = np.array(t_sync_list, dtype=np.float64)
    t_rerank = np.array(t_rerank_list, dtype=np.float64)
    t_total = np.array(t_total_list, dtype=np.float64)

    n_events = np.array(n_events_list, dtype=np.int64)
    n_valid = np.array(n_valid_list, dtype=np.int64)
    n_kpts = np.array(n_kpts_list, dtype=np.int64)

    n_frames = int(t_total.size)
    achieved_hz = (n_frames / wall_s) if wall_s > 0 else 0.0
    over_budget = int(np.sum(t_total > args.dt_ms))
    pct_over = 100.0 * over_budget / max(1, n_frames)

    print("\n========== SUMMARY ==========")
    print(f"frames (post-warmup): {n_frames}  (warmup skipped: {args.warmup})")
    print(f"wall time: {wall_s:.3f} s  => achieved: {achieved_hz:.2f} Hz   target: {args.target_hz:.2f} Hz")
    print(f"budget: {args.dt_ms:.2f} ms/window  | over-budget: {over_budget}/{n_frames} ({pct_over:.1f}%)\n")

    print(summarize_ms(t_read,   "read+window (CPU)"))
    print(summarize_ms(t_vit,    "vit path (GPU stream A)"))
    print(summarize_ms(t_ret,    "retrieval (GPU, inside A)"))
    print(summarize_ms(t_se,     "super path (GPU stream B)"))
    print(summarize_ms(t_join,   "joined GPU time (A||B overlap)"))
    print(summarize_ms(t_sync,   "sync time (CPU up to GPU done)"))
    print(summarize_ms(t_rerank, "rerank (CPU, topK only)"))
    print(summarize_ms(t_total,  "end2end total (CPU sync + rerank)"))
    print()
    if n_frames:
        print(f"events/window | mean {n_events.mean():.1f}  med {np.median(n_events):.1f}  "
              f"p95 {np.percentile(n_events,95):.1f}  max {n_events.max()}")
        print(f"valid/window  | mean {n_valid.mean():.1f}  med {np.median(n_valid):.1f}  "
              f"p95 {np.percentile(n_valid,95):.1f}  max {n_valid.max()}")
        print(f"kpts/window   | mean {n_kpts.mean():.1f}  med {np.median(n_kpts):.1f}  "
              f"p95 {np.percentile(n_kpts,95):.1f}  max {n_kpts.max()}")
    print("================================\n")


if __name__ == "__main__":
    main()
