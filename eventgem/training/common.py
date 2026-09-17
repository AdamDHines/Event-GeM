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
    # single-exponent and structural ablation slices (2026-08-25 paper ablations)
    "p1": slice(0, 128),          # GeM p=1 (average pooling) only, 128-D
    "p2": slice(128, 256),        # GeM p=2 only, 128-D
    "p3": slice(256, 384),        # GeM p=3 only, 128-D
    "p5_regional": slice(384, 1024),  # p=5 + 2x2 regional (no multi-exponent), 640-D
}
GEM_PS = [1.0, 2.0, 3.0, 5.0]

# --- v2 bank layout (2026-08-27 gem-exponent-factorial) -----------------------------------
# The v1 layout above fixes the regional blocks at p=5, so "regional multi-exponent" is not
# expressible as a slice of it. The v2 layout pools every exponent both globally and per
# quadrant, which makes the whole global-x-regional factorial a column selection again:
#
#   blocks 0..5    global GeM, p in {1,2,3,5,10,max}          6 x 128
#   blocks 6..25   2x2 regional GeM, p in {1,2,3,5,max}       5 x (4 x 128), p-major
#
# Combinations like g5+r5 are no longer contiguous, so v2 variants are lists of 128-D BLOCK
# IDS rather than slices. v1 banks, v1 heads and the 2026-08-25 artifacts are untouched: the
# names in DESC_SLICES and DESC_BLOCKS are disjoint, and prepare_input dispatches on which
# registry the variant is in.
BLOCK_DIM = 128
GEM_PS_V2 = [1.0, 2.0, 3.0, 5.0, 10.0, "max"]     # global blocks, in order
GEM_PS_V2_REGIONAL = [1.0, 2.0, 3.0, 5.0, "max"]  # regional blocks, in order
N_QUADRANTS = 4
DESC_DIM_V2 = (len(GEM_PS_V2) + len(GEM_PS_V2_REGIONAL) * N_QUADRANTS) * BLOCK_DIM  # 3328


def _v2_global(p):
    return [GEM_PS_V2.index(p)]


def _v2_regional(p):
    start = len(GEM_PS_V2) + GEM_PS_V2_REGIONAL.index(p) * N_QUADRANTS
    return list(range(start, start + N_QUADRANTS))


_G = {f"g{k}": _v2_global(p) for k, p in
      zip(("1", "2", "3", "5", "10", "max"), GEM_PS_V2)}
_R = {f"r{k}": _v2_regional(p) for k, p in
      zip(("1", "2", "3", "5", "max"), GEM_PS_V2_REGIONAL)}
_GMULTI = _G["g1"] + _G["g2"] + _G["g3"] + _G["g5"]
_GALL = sorted(sum(_G.values(), []))
_RMULTI = _R["r1"] + _R["r2"] + _R["r3"] + _R["r5"]
_RALL = sorted(sum(_R.values(), []))

DESC_BLOCKS = {
    **_G, **_R,
    "gmulti": _GMULTI,                          # == v1 "multi_p"
    "gall": _GALL,
    "rmulti": _RMULTI,
    "rall": _RALL,
    "g5_r5": _G["g5"] + _R["r5"],               # == v1 "p5_regional"
    "gmulti_r5": _GMULTI + _R["r5"],            # == v1 "regional" (today's full descriptor)
    "g5_rmulti": _G["g5"] + _RMULTI,
    "gmulti_rmulti": _GMULTI + _RMULTI,
    "gmax_rmax": _G["gmax"] + _R["rmax"],
    "gall_rall": _GALL + _RALL,
}

# The three v2 variants that must reproduce a v1 variant exactly (verify_v2.py asserts this).
V1_EQUIVALENTS = {"gmulti": "multi_p", "g5": "p5", "g5_r5": "p5_regional",
                  "gmulti_r5": "regional", "g1": "p1", "g2": "p2", "g3": "p3"}


# --- spatial pyramid layout (2026-08-28 pyramid-gem) ------------------------------------
# The 68-condition sweep showed the 2x2 regional blocks carry essentially all of the signal
# (+12.30 mean R@1 over p=5, winning 68/68 conditions) while multi-exponent on top of them is
# worth -0.45. So the live question is spatial, not spectral: if 2x2 is worth +12.3, is 4x4
# or 8x8 worth more? The pyramid pools one exponent at several grid resolutions.
#
# On the 30x40 FPN grid the levels are all well-formed -- 2x2 gives 15x20 cells, 4x4 gives
# 7-8 x 10, 8x8 gives 3-4 x 5. 16x16 would give 2x2-unit cells and is not included.
PYRAMID_LEVELS = (1, 2, 4, 8)
PYRAMID_P = 5.0
PYRAMID_DIM = sum(g * g for g in PYRAMID_LEVELS) * BLOCK_DIM      # 85 blocks, 10880-D


def _cell_bounds(n, g):
    """Cell boundaries for a g x g split of n units, tolerating uneven division.

    At level 1 and 2 this reproduces pool_descriptor's own splits exactly (whole map, and
    n//2), which is what lets the L12 sub-pyramid reproduce p5_regional bit for bit.
    """
    return np.linspace(0, n, g + 1).astype(int)


def pool_pyramid(feats, p=PYRAMID_P, levels=PYRAMID_LEVELS):
    """Spatial-pyramid GeM at a single exponent. feats: [B, 128, H, W] -> [B, 85*128].

    Same arithmetic as pool_descriptor's regional half, level by level: the float32
    max-rescale, and every cell divided by the GLOBAL per-(batch, channel) max rather than
    its own, so cells keep a common scale reference and encode *where* rather than *what*.
    Cells are emitted row-major within each level, so level 2 is TL, TR, BL, BR -- the order
    pool_descriptor uses.
    """
    x = feats.clamp(min=1e-6)
    m = x.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    xn = x / m
    H, W = x.shape[-2], x.shape[-1]
    parts = []
    for g in levels:
        ys, xs = _cell_bounds(H, g), _cell_bounds(W, g)
        for i in range(g):
            for j in range(g):
                q = xn[..., ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                pooled = F.avg_pool2d(q.pow(p), (q.shape[-2], q.shape[-1])).pow(1.0 / p)
                parts.append((pooled * m).flatten(1))
    return torch.cat(parts, dim=1)


def _pyramid_block_ids():
    """level -> the block ids it occupies in a pool_pyramid bank."""
    out, start = {}, 0
    for g in PYRAMID_LEVELS:
        out[g] = list(range(start, start + g * g))
        start += g * g
    return out


_PY = _pyramid_block_ids()
DESC_BLOCKS.update({
    "L1": _PY[1],                                        # global only, 128-D
    "L2": _PY[2],                                        # 2x2 only, 512-D
    "L4": _PY[4],                                        # 4x4 only, 2048-D
    "L8": _PY[8],                                        # 8x8 only, 8192-D
    "L12": _PY[1] + _PY[2],                              # == v1 p5_regional, 640-D
    "L124": _PY[1] + _PY[2] + _PY[4],                    # 2688-D
    "L1248": _PY[1] + _PY[2] + _PY[4] + _PY[8],          # 10880-D
    "L14": _PY[1] + _PY[4],                              # skip 2x2: is it redundant? 2176-D
    "L18": _PY[1] + _PY[8],                              # 8320-D
    "L148": _PY[1] + _PY[4] + _PY[8],                    # 10368-D
})
# The pyramid variant that must reproduce a v1 variant exactly (verify_pyramid.py asserts it).
PYRAMID_V1_EQUIVALENT = {"L12": "p5_regional", "L1": "p5"}


def variant_dim(variant):
    """Descriptor width for a variant name from either registry."""
    if variant in DESC_BLOCKS:
        return len(DESC_BLOCKS[variant]) * BLOCK_DIM
    s = DESC_SLICES[variant]
    return s.stop - s.start

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


def _quadrants(H, W):
    """The 2x2 split pool_descriptor uses, in TL, TR, BL, BR order."""
    h2, w2 = H // 2, W // 2
    return ((slice(None, h2), slice(None, w2)), (slice(None, h2), slice(w2, None)),
            (slice(h2, None), slice(None, w2)), (slice(h2, None), slice(w2, None)))


def _gem_block(xn, m, p):
    """One 128-D pooled block from the rescaled map. `p` is a float exponent or "max".

    Written to match pool_descriptor's arithmetic operation-for-operation so the shared
    blocks of a v2 bank reproduce a v1 bank exactly (verify_v2.py asserts cosine 1.000000).
    The rescale by the per-(batch, channel) max is what keeps p=10 finite in float32.
    """
    if p == "max":
        return (xn.amax(dim=(-2, -1), keepdim=True) * m).flatten(1)
    pooled = F.avg_pool2d(xn.pow(p), (xn.shape[-2], xn.shape[-1])).pow(1.0 / p)
    return (pooled * m).flatten(1)


def pool_descriptor_v2(feats):
    """The v2 superset descriptor for a trunk feature map. feats: [B, 128, H, W] float32.

    [ global GeM p in {1,2,3,5,10,max} | 2x2 regional GeM p in {1,2,3,5,max} ] -> [B, 3328],
    laid out as DESC_BLOCKS documents. Every v1 variant is a block selection of this, so the
    v1 bank remains derivable and the global-x-regional exponent factorial becomes free.
    """
    x = feats.clamp(min=1e-6)
    m = x.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    xn = x / m
    parts = [_gem_block(xn, m, p) for p in GEM_PS_V2]
    for p in GEM_PS_V2_REGIONAL:
        for ys, xs in _quadrants(x.shape[-2], x.shape[-1]):
            parts.append(_gem_block(xn[..., ys, xs], m, p))
    out = torch.cat(parts, dim=1)
    assert out.shape[1] == DESC_DIM_V2
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
