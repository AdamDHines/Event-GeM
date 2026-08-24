"""Shared pieces for the NYC-Event contrastive head: recording discovery, GPS handling,
pooling, and the descriptor-bank layout.

The bank layout is the contract between extract_banks.py and train_head.py: each frame's
descriptor is the concatenation of GeM poolings of the frozen SuperEvent trunk's feature map,

    [ GeM p=1 (128) | GeM p=2 (128) | GeM p=3 (128) | GeM p=5 (128) | 2x2 regional GeM p=5 (512) ]

so a bank is float32 [K, 1024] and the training-time --input variants are column slices of it.
"""
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DESC_DIM = 1024
DESC_SLICES = {
    "p5": slice(384, 512),        # production descriptor (GeM p=5), 128-D
    "multi_p": slice(0, 512),     # GeM p in {1,2,3,5}, 512-D
    "regional": slice(0, 1024),   # multi_p + 2x2 regional p=5, 1024-D
}
GEM_PS = [1.0, 2.0, 3.0, 5.0]

# Brisbane framing offsets (ms, from pixi.toml) and known bank row counts on current data.
BRISBANE_OFFSETS = {
    "sunset2": 1587540271650, "sunset1": 1587452582350, "morning": 1588029265730,
    "daytime": 1587705130800, "sunrise": 1588105232910, "night": 1587975221100,
}
BRISBANE_ROWS = {"sunset2": 12825, "sunset1": 14478, "morning": 13453, "daytime": 14318}

# All 15 NYC recordings train (campaign decision 2026-08-23: NYC is the training corpus,
# not a benchmark — selection happens on the Brisbane sunset1 monitor). These pairs are
# retained as DIAGNOSTIC retrieval metrics only (highest measured route overlap: 56.1% and
# 12.9%); they select nothing and their recordings are in the training set. NB the
# dataset's own GT pairing (17-58-34 vs 13-59-10) shares <3% of its route — unusable.
NYC_DIAG_PAIRS = [
    ("2023-02-14_18-20-40", "2023-02-14_15-06-30"),   # night query -> day reference
    ("2022-12-09_18-56-13", "2022-12-09_19-42-07"),   # night, night
]
NYC_SENSOR_SIZE = (1280, 720)  # (W, H), passed to ecv.open to skip the coordinate scan


# megaevent's prepared NYC-Event-VPR samples (scripts/prepare_nyc_event.py in that repo):
# 33.333 ms event windows at 1 Hz, cut straight from the RAW EVT3 archives whose headers
# carry the absolute recording epoch — so every sample's GPS fix is exact by construction
# (gps_delta gated at 550 ms). The hdf5 conversions of this dataset dropped those headers
# and concatenated multi-segment recordings, which is why alignment could not be recovered
# from them (measured 2026-08-23: anchors wrong by +85 s and worse; four files unsorted).
MEGAEVENT_NYC = "/media/adam/vprdatasets/megaevent/nycevent"


def discover_nyc_recordings(nyc_root=MEGAEVENT_NYC):
    """All NYC traverses as sorted (name, rows) pairs from megaevent's manifest.

    Each row: {"path": npz path, "lat": float, "lon": float, "heading": float (degrees),
    "t_wall_s": float}, sorted by wall-clock time within the traverse. The npz files hold
    x/y/t/p (t in relative µs, p in ±1) plus `resolution`.
    """
    import csv as _csv
    manifest = Path(nyc_root) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found — run megaevent/scripts/prepare_nyc_event.py first")
    groups = {}
    with open(manifest, newline="") as handle:
        for row in _csv.DictReader(handle):
            groups.setdefault(row["traverse"], []).append({
                "path": row["path"],
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "heading": float(row["heading"]),
                "t_wall_s": int(row["sample_wall_time_us"]) / 1e6,
            })
    for rows in groups.values():
        rows.sort(key=lambda r: r["t_wall_s"])
    return sorted(groups.items())


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres; accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pool_descriptor(feats):
    """The bank-layout descriptor for a trunk feature map. feats: [B, 128, H, W] float32.

    GeM uses the production clamp with the float32 rescale from sweep_gem_p.gem_multi; the
    regional half splits the map into 2x2 and GeM(p=5)-pools each quadrant.
    """
    x = feats.clamp(min=1e-6)
    m = x.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    xn = x / m
    parts = []
    for p in GEM_PS:
        pooled = F.avg_pool2d(xn.pow(p), (x.shape[-2], x.shape[-1])).pow(1.0 / p)
        parts.append((pooled * m).flatten(1))
    h2, w2 = x.shape[-2] // 2, x.shape[-1] // 2
    for ys, xs in ((slice(None, h2), slice(None, w2)), (slice(None, h2), slice(w2, None)),
                   (slice(h2, None), slice(None, w2)), (slice(h2, None), slice(w2, None))):
        q, mq = xn[..., ys, xs], m
        pooled = F.avg_pool2d(q.pow(5.0), (q.shape[-2], q.shape[-1])).pow(1.0 / 5.0)
        parts.append((pooled * mq).flatten(1))
    out = torch.cat(parts, dim=1)
    assert out.shape[1] == DESC_DIM
    return out


def crop_offsets(H, W, se_config):
    """The centre-crop EventGeMMCTS applies so H, W divide the trunk's downsample factor.
    Returns (off_top, h_end, off_left, w_end); a no-op crop returns (0, H, 0, W)."""
    max_factor = se_config["grid_size"]
    if "backbone_config" in se_config:
        stage_blocks = se_config["backbone_config"]["num_blocks"]
        patch = se_config["backbone_config"]["stem"]["patch_size"]
        max_factor = patch * (2 ** (len(stage_blocks) - 1))
        if "attention" in se_config["backbone_config"]["stage"]:
            max_factor *= int(np.max(
                se_config["backbone_config"]["stage"]["attention"]["partition_size"]))
    crop = np.array([H, W]) % max_factor
    off_top, off_bottom = math.ceil(crop[0] / 2), math.floor(crop[0] / 2)
    off_left, off_right = math.ceil(crop[1] / 2), math.floor(crop[1] / 2)
    return off_top, H - off_bottom if off_bottom else H, off_left, W - off_right if off_right else W


def recall_at_k(sim, gt, ks):
    """Recall@K on a similarity matrix, rows = references, cols = queries — the repo's
    recallAtK convention (full argsort, best last; queries with no GT match dropped)."""
    valid = gt.sum(axis=0) > 0
    if not valid.any():
        return {k: float("nan") for k in ks}
    s, g = sim[:, valid], gt[:, valid]
    order = np.argsort(s, axis=0)
    out = {}
    for k in ks:
        hits = np.take_along_axis(g, order[-k:, :], axis=0).any(axis=0)
        out[k] = float(hits.mean())
    return out


def atomic_save(obj, path):
    """torch.save through a temp file + os.replace so a killed job never leaves a torn file."""
    path = str(path)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)
