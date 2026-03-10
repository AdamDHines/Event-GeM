from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import math
import yaml
import os
import torch
import eventgem.streamutils.stream as stream
# ---------- HDF5 helpers ----------


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

NS_PER_S = 1_000_000_000

def sec_to_ns(t_sec: float) -> int:
    return int(round(float(t_sec) * NS_PER_S))

def ns_to_sec(t_ns: int) -> float:
    return float(t_ns) / NS_PER_S

def binary_search_event_index_ns(t_dset, target_ns: int) -> int:
    """
    Binary search on integer nanosecond timestamps (epoch ns).
    Returns index closest to (but not exceeding) target_ns.
    Returns 0 if target is before first event, N if after last event.
    """
    N = len(t_dset)
    if N == 0:
        return 0

    t_first = int(t_dset[0])
    if target_ns <= t_first:
        return 0

    t_last = int(t_dset[-1])
    if target_ns >= t_last:
        return N

    low, high = 0, N - 1
    best_idx = 0
    while low <= high:
        mid = (low + high) // 2
        t_mid = int(t_dset[mid])
        if t_mid < target_ns:
            best_idx = mid
            low = mid + 1
        else:
            high = mid - 1
    return best_idx

def binary_search_event_index(t_dset, target_time_sec, time_scale):
    """
    Perform binary search on the HDF5 timestamp dataset to find the 
    index closest to (but not exceeding) target_time_sec.
    Returns 0 if target is before the first event, N if after the last event.
    """
    N = len(t_dset)
    if N == 0:
        return 0

    low = 0
    high = N - 1

    t_first = float(t_dset[0]) * time_scale
    if target_time_sec <= t_first:
        return 0

    t_last = float(t_dset[-1]) * time_scale
    if target_time_sec >= t_last:
        return N

    best_idx = 0

    while low <= high:
        mid = (low + high) // 2
        t_mid = float(t_dset[mid]) * time_scale

        if t_mid < target_time_sec:
            best_idx = mid  # candidate: we want the index just before the target time
            low = mid + 1
        else:
            high = mid - 1

    return best_idx


# ---------- MCTS core ----------

def mcts_numpy(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    t_ref: float,
    time_windows: np.ndarray,
):
    """
    NumPy implementation of the MCTS encoding from SuperEvent.

    - Input events (x,y,t,p) are expected to contain all events relevant to the
      current time surface (i.e., events in [t_ref-DeltaT_max, t_ref)).
    - The output MCTS is computed for the reference time t_ref.
    """
    assert x.ndim == y.ndim == t.ndim == p.ndim == 1
    num_scales = len(time_windows)
    C = 2 * num_scales
    mcts = np.zeros((C, height, width), dtype=np.float32)

    if x.size == 0:
        return mcts

    # Ensure types
    x = x.astype(np.int64)
    y = y.astype(np.int64)
    t = t.astype(np.float64)
    p = p.astype(np.int8)

    # Clamp coordinates to image bounds
    valid_xy = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid_xy], y[valid_xy], t[valid_xy], p[valid_xy]

    if x.size == 0:
        return mcts

    # # Sort by time (ascending)
    # order = np.argsort(t)
    # x, y, t, p = x[order], y[order], t[order], p[order]

    # Split by polarity: p>0 -> positive, p<=0 -> negative
    for pol_idx, pol_sign in enumerate((1, -1)):
        if pol_sign > 0:
            mask_pol = p > 0
        else:
            mask_pol = p <= 0

        if not np.any(mask_pol):
            continue

        x_p, y_p, t_p = x[mask_pol], y[mask_pol], t[mask_pol]

        # Precompute flattened indices for the relevant polarity events
        idx_flat_all = y_p * width + x_p

        for s_idx, DeltaT in enumerate(time_windows):
            DeltaT = float(DeltaT)
            window_start = t_ref - DeltaT

            # Events inside [t_ref - ΔT, t_ref]
            in_win = t_p >= window_start

            if not np.any(in_win):
                continue

            idx_flat = idx_flat_all[in_win]
            t_valid = t_p[in_win]

            # Reverse so np.unique gives last occurrence in time
            idx_rev = idx_flat[::-1]
            t_rev = t_valid[::-1]

            unique_pix, first_idx_rev = np.unique(idx_rev, return_index=True)

            # Convert reversed index back to forward index
            last_idx = len(idx_rev) - 1 - first_idx_rev
            t_last = t_valid[last_idx]

            # Recover (y,x) from flattened index
            y_last = (unique_pix // width).astype(np.int64)
            x_last = (unique_pix % width).astype(np.int64)

            # Time Surface value: exp( -Δt / ΔT )
            dt = t_ref - t_last
            dt = np.clip(dt, 0.0, DeltaT)

            values = np.exp(-dt / DeltaT).astype(np.float32)

            ch = pol_idx * num_scales + s_idx
            mcts[ch, y_last, x_last] = values

    return mcts


# ---------- Single-file MCTS generator ----------

@torch.no_grad()
def gpu_mcts(h5_path,
            out_dir,
            time_windows,
            time_scale,
            H,
            W,
            chunk_size,
            start_time_sec,
            skip=0):
    
    os.makedirs(out_dir, exist_ok=True)
    event_iter = stream.stream_event_windows_raw(
        h5_path, time_windows[-1], chunk_size, time_scale, start_time_sec, skip
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows_sec = torch.tensor(np.array(time_windows, dtype=np.float32) * 1e-3, device=device)
    S = int(windows_sec.numel())
    flat = H * W
    for (_, _, t_ref_raw, x, y, t, p, frame_idx, _) in event_iter:
        out = torch.zeros((2 * S, H, W), device=device, dtype=torch.float32)

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        t = torch.from_numpy(t).to(device)
        p = torch.from_numpy(p).to(device)

        # Move to GPU (still expensive; see section 3)

        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if not torch.any(valid):
            return out

        x = x[valid]; y = y[valid]; t = t[valid]; p = p[valid]

        # Relative time (seconds, negative = past)
        t_rel = (t - int(t_ref_raw)).to(torch.float32) * float(time_scale)

        # Only keep events within max window (saves work)
        Dt_max = float(windows_sec.max().item())
        m = t_rel >= -Dt_max
        if not torch.any(m):
            return out
        x = x[m]; y = y[m]; p = p[m]; t_rel = t_rel[m]

        lin = y * W + x  # int32
        # polarity group: 0 for pos, 1 for neg (match your earlier convention)
        grp = torch.where(p > 0, torch.zeros_like(lin), torch.ones_like(lin))
        idx = lin + grp * flat  # [0..2*flat)

        # One scatter_reduce for "last timestamp per pixel per polarity"
        neg_inf = -1e20
        t_last = torch.full((2 * flat,), neg_inf, device=device, dtype=torch.float32)
        t_last.scatter_reduce_(0, idx.to(torch.int64), t_rel, reduce="amax", include_self=True)

        # reshape to [2,H,W]
        t_last = t_last.view(2, H, W)
        valid_pix = t_last > -1e10

        dt = (-t_last).clamp(min=0.0)  # [2,H,W], seconds since last event

        # Vectorize over S windows: produce [S,2,H,W]
        Dt = windows_sec.to(device=device, dtype=torch.float32).view(S, 1, 1, 1)          # [S,1,1,1]
        dt_s = dt.unsqueeze(0).expand(S, -1, -1, -1)                                     # [S,2,H,W]
        mask = valid_pix.unsqueeze(0) & (dt_s <= Dt)                                     # [S,2,H,W]

        vals = torch.where(mask, torch.exp(-dt_s / Dt), torch.zeros_like(dt_s))          # [S,2,H,W]

        # Pack to your output layout: [S,H,W] pos then [S,H,W] neg
        out[:S] = vals[:, 0]
        out[S:] = vals[:, 1]

        # 4. Save and step forward
        out_path = out_dir / f"mcts_{frame_idx:05d}.npz"
        np.savez_compressed(out_path, mcts=out.detach().cpu().numpy())

        frame_idx += 1


# ---------- Public entry: ref + query ----------

def gen_mcts(
    root,
    dataset,
    reference,
    query,
    mcts_time,
    chunk_size=500_000,
    max_frames=None,
    use_event_counts: bool = False,  # <-- NEW FLAG
):
    root = Path(root)
   # time_windows = np.array(mcts_time, dtype=np.float64)
    # convert to seconds from msec
   # time_windows = time_windows * 1e-3

    ref_path = root / dataset / reference / f"{reference}.hdf5"
    query_path = root / dataset / query / f"{query}.hdf5"

    # NOTE: keeping your original out_dir behaviour (same dir for ref and query)
    ref_dir = root / dataset / reference / f"mcts_{reference}_{mcts_time[-1]}"
    qry_dir = root / dataset / query / f"mcts_{query}_{mcts_time[-1]}"

    # Time scale and sensor size
    if dataset == "brisbane_event":
        time_scale = 1e-9  # (not used in epoch-ns branch, but keep)
        t_is_epoch_ns = True
        H, W = 240, 346
        config_path = "./eventgem/external/eventlab/datasets/brisbane_event.yaml"
        config = yaml.safe_load(open(config_path, 'r'))
        ref_start = config['other']['offset'][reference]
        query_start = config['other']['offset'][query]
    elif dataset == "nsavp":
        time_scale = 1e-9  # nanoseconds to seconds
        ref_start = 0.0
        query_start = 0.0
        H, W = 480, 640
        t_is_epoch_ns = True
    elif dataset == "fast_slow" or dataset == "qcr_event":
        time_scale = 1e-6  # microseconds to seconds
        ref_start = 0.0
        query_start = 0.0
        H, W = 240, 346
        t_is_epoch_ns = False
    else:
        raise NotImplementedError(f"Dataset not supported for MCTS generation: {dataset}")
    
    # Reference sequence
    if not os.path.exists(ref_dir):
        gpu_mcts(
            h5_path=ref_path,
            out_dir=ref_dir,
            time_windows=mcts_time,
            time_scale=time_scale,
            H=H,
            W=W,
            chunk_size=chunk_size,
            start_time_sec=ref_start
        )

    # Query sequence
    if not os.path.exists(qry_dir):
        gpu_mcts(
            h5_path=query_path,
            out_dir=qry_dir,
            time_windows=mcts_time,
            time_scale=time_scale,
            H=H,
            W=W,
            chunk_size=chunk_size,
            max_frames=max_frames,
            start_time_sec=query_start,
            use_event_counts=use_event_counts,
            t_is_epoch_ns=t_is_epoch_ns
        )
