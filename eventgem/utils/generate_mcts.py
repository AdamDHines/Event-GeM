import argparse
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import math

# ---------- HDF5 helpers ----------

def find_event_datasets(f: h5py.File):
    """Return (x, y, t, p) datasets from an HDF5 file with group 'events'."""
    if "events" not in f or not isinstance(f["events"], h5py.Group):
        raise RuntimeError("Expected a group 'events' in the HDF5 file.")
    g = f["events"]
    for key in ["x", "y", "t", "p"]:
        if key not in g:
            raise RuntimeError(f"Expected dataset 'events/{key}' in the HDF5 file.")
    return g["x"], g["y"], g["t"], g["p"]


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

    # Sort by time (ascending)
    order = np.argsort(t)
    x, y, t, p = x[order], y[order], t[order], p[order]

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

def _generate_mcts_for_file(
    h5_path: Path,
    out_dir: Path,
    time_windows: np.ndarray,
    time_scale: float,
    H: int,
    W: int,
    chunk_size: int = 500_000,
    max_frames: int | None = None,
    start_time_sec: float = 0.0,
):
    """Generate MCTS frames for a single HDF5 file."""
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    DT_MAX = float(time_windows[-1])  # max window size in seconds

    with h5py.File(h5_path, "r") as f:
        x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)
        N = len(t_dset)

        # 1. Determine Start Index via Binary Search (efficient seek)
        seek_idx = binary_search_event_index(t_dset, start_time_sec, time_scale)
        read_idx = seek_idx + 1

        if read_idx >= N:
            print(f"[WARN] Start time is beyond the end of the file or file is empty: {h5_path}")
            return

        # 2. Setup Time Tracking
        t_first_raw = t_dset[0]
        t_last_raw = t_dset[N - 1]
        t_start_sec = float(t_first_raw) * time_scale
        t_end_sec = float(t_last_raw) * time_scale

        first_t_ref_candidate = max(t_start_sec + DT_MAX, start_time_sec + DT_MAX)

        if t_end_sec > first_t_ref_candidate:
            num_steps = math.ceil((first_t_ref_candidate - t_start_sec) / DT_MAX)
            current_t_ref = t_start_sec + num_steps * DT_MAX
        else:
            current_t_ref = t_end_sec  # Fallback if range is tiny

        print(f"[INFO] File: {h5_path}")
        print(f"[INFO] Time Range (s): [{t_start_sec:.3f}, {t_end_sec:.3f}]")
        print(f"[INFO] Seek Index: {seek_idx}. Reading from event index: {read_idx}")
        print(f"[INFO] First MCTS frame t_ref: {current_t_ref:.3f} s")
        print(f"[INFO] MCTS window ΔT_max = {DT_MAX*1000:.1f} ms")

        # 3. Estimation and TQDM setup
        if t_end_sec > current_t_ref:
            est_frames = int(np.ceil((t_end_sec - current_t_ref) / DT_MAX)) + 1
        else:
            est_frames = 1

        total_frames = est_frames if max_frames is None else min(est_frames, max_frames)
        print(f"[INFO] Estimated frames: {est_frames}, generating: {total_frames}")

        # Buffers for the full relevant history [t_ref - DT_MAX, current_t_ref)
        buf_x = np.empty(0, dtype=np.int64)
        buf_y = np.empty(0, dtype=np.int64)
        buf_t = np.empty(0, dtype=np.float64)
        buf_p = np.empty(0, dtype=np.int8)

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc=f"MCTS frames ({h5_path.name})")

        # --- Main Streaming Loop ---
        while current_t_ref < t_end_sec and frame_idx < total_frames:
            t_history_start = current_t_ref - DT_MAX

            # 1. Fill buffer until we have events past current_t_ref (or hit EOF)
            while (buf_t.size == 0 or buf_t[-1] < current_t_ref) and read_idx < N:
                end_idx = min(N, read_idx + chunk_size)

                c_x = x_dset[read_idx:end_idx].astype(np.int64)
                c_y = y_dset[read_idx:end_idx].astype(np.int64)
                c_t = t_dset[read_idx:end_idx].astype(np.float64) * time_scale
                c_p = p_dset[read_idx:end_idx].astype(np.int8)

                if buf_t.size == 0:
                    buf_x, buf_y, buf_t, buf_p = c_x, c_y, c_t, c_p
                else:
                    buf_x = np.concatenate([buf_x, c_x])
                    buf_y = np.concatenate([buf_y, c_y])
                    buf_t = np.concatenate([buf_t, c_t])
                    buf_p = np.concatenate([buf_p, c_p])

                read_idx = end_idx

            # 2. Prune buffer and extract window [t_history_start, current_t_ref)
            if buf_t.size > 0:
                keep_mask = (buf_t >= t_history_start) & (buf_t < current_t_ref)

                x_win = buf_x[keep_mask]
                y_win = buf_y[keep_mask]
                t_win = buf_t[keep_mask]
                p_win = buf_p[keep_mask]

                # Keep only events >= t_history_start for the next iteration
                keep_next_loop = buf_t >= t_history_start
                buf_x = buf_x[keep_next_loop]
                buf_y = buf_y[keep_next_loop]
                buf_t = buf_t[keep_next_loop]
                buf_p = buf_p[keep_next_loop]
            else:
                x_win = y_win = t_win = p_win = np.empty(0)

            # 3. Compute MCTS
            mcts = mcts_numpy(
                x_win,
                y_win,
                t_win,
                p_win,
                height=H,
                width=W,
                t_ref=current_t_ref,
                time_windows=time_windows,
            )

            # 4. Save and step forward
            out_path = out_dir / f"mcts_{frame_idx:05d}.npz"
            np.savez_compressed(out_path, mcts=mcts, t_ref=current_t_ref)

            frame_idx += 1
            current_t_ref += DT_MAX
            pbar.update(1)

        pbar.close()
        print(f"[INFO] Completed. Generated {frame_idx} MCTS frames into {out_dir}")


# ---------- Public entry: ref + query ----------

def gen_mcts(root, dataset, reference, query, mcts_time, chunk_size=500_000, max_frames=None):
    root = Path(root)
    time_windows = np.array(mcts_time, dtype=np.float64)
    DT_MAX = float(time_windows[-1])

    ref_path = root / dataset / f"{reference}.hdf5"
    query_path = root / dataset / f"{query}.hdf5"

    # NOTE: keeping your original out_dir behaviour (same dir for ref and query)
    out_dir = root / dataset / f"mcts_{reference}"

    # Time scale and sensor size
    if dataset == "brisbane_event":
        time_scale = 1e-9  # nanoseconds to seconds
        H, W = 240, 346
    else:
        time_scale = 1e-6  # microseconds to seconds
        # TODO: set H, W for other datasets
        raise ValueError(f"Unknown dataset {dataset}, please set H/W and time_scale.")

    # Reference sequence
    _generate_mcts_for_file(
        h5_path=ref_path,
        out_dir=out_dir,
        time_windows=time_windows,
        time_scale=time_scale,
        H=H,
        W=W,
        chunk_size=chunk_size,
        max_frames=max_frames,
        start_time_sec=0.0,
    )

    # Query sequence
    _generate_mcts_for_file(
        h5_path=query_path,
        out_dir=out_dir,
        time_windows=time_windows,
        time_scale=time_scale,
        H=H,
        W=W,
        chunk_size=chunk_size,
        max_frames=max_frames,
        start_time_sec=0.0,
    )
