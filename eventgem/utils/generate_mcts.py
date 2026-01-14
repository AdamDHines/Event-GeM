from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
import math
import yaml
import os

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

def _generate_mcts_for_file(
    h5_path: Path,
    out_dir: Path,
    time_windows: np.ndarray,
    time_scale: float,
    H: int,
    W: int,
    chunk_size: int = 500_000,
    max_frames = None,
    start_time_sec: float = 0.0,
    use_event_counts: bool = False,   # <-- NEW FLAG
):
    """Generate MCTS frames for a single HDF5 file.

    If use_event_counts is False:
        - Behaves as before: time_windows are in seconds, frames step by DT_MAX.

    If use_event_counts is True:
        - time_windows are interpreted as event counts.
        - The largest window (time_windows[-1]) is the number of events per frame.
        - Frames are generated every E_MAX events, starting after start_time_sec.
        - 'Time' in mcts_numpy is replaced by event indices.
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)
        N = len(t_dset)

        if N == 0:
            print(f"[WARN] No events in file: {h5_path}")
            return

        # -----------------------------
        # EVENT-COUNT-BASED MODE
        # -----------------------------
        if use_event_counts:
            event_windows = np.array(time_windows, dtype=np.int64)
            if np.any(event_windows <= 0):
                raise ValueError(f"All event windows must be > 0 in event-count mode, got: {event_windows}")

            E_MAX = int(event_windows[-1])
            if E_MAX <= 0:
                raise ValueError("Largest event window (E_MAX) must be > 0.")

            # Respect start_time_sec by converting it to an event index
            seek_idx = binary_search_event_index(t_dset, start_time_sec, time_scale)
            read_idx = seek_idx + 1  # start just AFTER the start_time_sec

            if read_idx >= N:
                print(f"[WARN] Start time is beyond the end of the file or file is empty: {h5_path}")
                return

            first_idx = read_idx
            remaining_events = N - first_idx
            if remaining_events < E_MAX:
                print(
                    f"[WARN] Not enough events after start_time_sec for even one MCTS frame "
                    f"(needed {E_MAX}, have {remaining_events}) in file: {h5_path}"
                )
                return

            # Step size in events between frames (analogous to stepping by DT_MAX in time)
            step_events = E_MAX

            # Estimate number of frames
            est_frames = (remaining_events - E_MAX) // step_events + 1
            total_frames = est_frames if max_frames is None else min(est_frames, max_frames)

            print(f"[INFO] File: {h5_path}")
            print(f"[INFO] Mode: EVENT-COUNT-BASED")
            print(f"[INFO] Total events: {N}, starting at index {first_idx}")
            print(f"[INFO] Event windows: {event_windows}")
            print(f"[INFO] E_MAX (events per frame): {E_MAX}")
            print(f"[INFO] Estimated frames: {est_frames}, generating: {total_frames}")

            # Streaming buffers for x, y, p
            buf_x = np.empty(0, dtype=np.int64)
            buf_y = np.empty(0, dtype=np.int64)
            buf_p = np.empty(0, dtype=np.int8)

            buffer_start_idx = first_idx  # global index of buf_x[0]
            read_idx = first_idx

            # First frame reference index (global)
            ref_idx = first_idx + E_MAX - 1

            frame_idx = 0
            pbar = tqdm(total=total_frames, desc=f"MCTS frames ({h5_path.name}) [events]")

            while frame_idx < total_frames:
                # 1. Make sure we have all events up to ref_idx in the buffer
                while (buffer_start_idx + buf_x.size - 1) < ref_idx and read_idx < N:
                    end_idx = min(N, read_idx + chunk_size)

                    c_x = x_dset[read_idx:end_idx].astype(np.int64)
                    c_y = y_dset[read_idx:end_idx].astype(np.int64)
                    c_p = p_dset[read_idx:end_idx].astype(np.int8)

                    if buf_x.size == 0:
                        buf_x, buf_y, buf_p = c_x, c_y, c_p
                        buffer_start_idx = read_idx
                    else:
                        buf_x = np.concatenate([buf_x, c_x])
                        buf_y = np.concatenate([buf_y, c_y])
                        buf_p = np.concatenate([buf_p, c_p])

                    read_idx = end_idx

                # If we still don't have enough, we are done
                if (buffer_start_idx + buf_x.size - 1) < ref_idx:
                    print(f"[INFO] Reached end of file earlier than expected at frame {frame_idx}.")
                    break

                # 2. Define the window [win_start_global, win_end_global] (inclusive)
                win_end_global = ref_idx
                win_start_global = win_end_global - (E_MAX - 1)

                if win_start_global < buffer_start_idx:
                    # This shouldn't normally happen if we prune correctly, but guard anyway
                    win_start_global = buffer_start_idx
                    # Adjust to maintain E_MAX length if possible
                    win_end_global = min(win_start_global + E_MAX - 1, buffer_start_idx + buf_x.size - 1)

                start_local = win_start_global - buffer_start_idx
                end_local = win_end_global - buffer_start_idx

                x_win = buf_x[start_local:end_local + 1]
                y_win = buf_y[start_local:end_local + 1]
                p_win = buf_p[start_local:end_local + 1]

                # Pseudo-time = event indices, used as "t"
                t_win = np.arange(win_start_global, win_end_global + 1, dtype=np.float64)
                t_ref = float(win_end_global)

                # 3. Compute MCTS using event-index "time"
                mcts = mcts_numpy(
                    x_win,
                    y_win,
                    t_win,
                    p_win,
                    height=H,
                    width=W,
                    t_ref=t_ref,
                    time_windows=event_windows.astype(np.float64),
                )

                # 4. Save and step forward in event index
                out_path = out_dir / f"mcts_{frame_idx:05d}.npz"
                np.savez_compressed(out_path, mcts=mcts, t_ref=t_ref)

                frame_idx += 1
                pbar.update(1)

                # Next frame
                ref_idx += step_events
                if ref_idx >= N:
                    # No more full windows
                    break

                # 5. Prune buffer events that are no longer needed:
                #    for the *next* window, earliest index will be ref_idx - (E_MAX - 1)
                next_win_start_global = ref_idx - (E_MAX - 1)
                if next_win_start_global > buffer_start_idx:
                    drop = next_win_start_global - buffer_start_idx
                    if drop > 0:
                        buf_x = buf_x[drop:]
                        buf_y = buf_y[drop:]
                        buf_p = buf_p[drop:]
                        buffer_start_idx = next_win_start_global

            pbar.close()
            print(f"[INFO] Completed (event-count mode). Generated {frame_idx} MCTS frames into {out_dir}")
            return

        # -----------------------------
        # TIME-BASED MODE (ORIGINAL)
        # -----------------------------
        DT_MAX = float(time_windows[-1])  # max window size in seconds

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
        print(f"[INFO] Mode: TIME-BASED")
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
        pbar = tqdm(total=total_frames, desc=f"MCTS frames ({h5_path.name}) [time]")

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
        print(f"[INFO] Completed (time-based mode). Generated {frame_idx} MCTS frames into {out_dir}")


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
    time_windows = np.array(mcts_time, dtype=np.float64)
    DT_MAX = float(time_windows[-1])

    ref_path = root / dataset / reference / f"{reference}.hdf5"
    query_path = root / dataset / query / f"{query}.hdf5"

    # NOTE: keeping your original out_dir behaviour (same dir for ref and query)
    ref_dir = root / dataset / reference / f"mcts_{reference}_{int(mcts_time[-1]/1000)}"
    qry_dir = root / dataset / query / f"mcts_{query}_{int(mcts_time[-1]/1000)}"

    # Time scale and sensor size
    if dataset == "brisbane_event":
        time_scale = 1e-9  # nanoseconds to seconds
        H, W = 240, 346
        # load the event lab config for the brisbane event to get the start time
        config_path = "./eventgem/external/eventlab/datasets/brisbane_event.yaml"
        config = yaml.safe_load(open(config_path, 'r'))
        ref_start = config['other']['offset'][reference]
        query_start = config['other']['offset'][query]
    elif dataset == "nsavp":
        time_scale = 1e-9  # nanoseconds to seconds
        ref_start = 0.0
        query_start = 0.0
        H, W = 480, 640
    elif dataset == "fast_slow" or dataset == "qcr_event":
        time_scale = 1  # microseconds to seconds
        ref_start = 0.0
        query_start = 0.0
        H, W = 240, 346
    else:
        raise NotImplementedError(f"Dataset not supported for MCTS generation: {dataset}")
    
    # Reference sequence
    if not os.path.exists(ref_dir):
        _generate_mcts_for_file(
            h5_path=ref_path,
            out_dir=ref_dir,
            time_windows=time_windows,
            time_scale=time_scale,
            H=H,
            W=W,
            chunk_size=chunk_size,
            max_frames=max_frames,
            start_time_sec=ref_start,
            use_event_counts=use_event_counts,  # <-- PASS FLAG
        )

    # Query sequence
    if not os.path.exists(qry_dir):
        _generate_mcts_for_file(
            h5_path=query_path,
            out_dir=qry_dir,
            time_windows=time_windows,
            time_scale=time_scale,
            H=H,
            W=W,
            chunk_size=chunk_size,
            max_frames=max_frames,
            start_time_sec=query_start,
            use_event_counts=use_event_counts,  # <-- PASS FLAG
        )
