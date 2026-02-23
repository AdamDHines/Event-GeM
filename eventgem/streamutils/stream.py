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
import sinabs.layers as sl
from tqdm import tqdm
import logging
from streamutils.EventVLAD import Imagenet_vgg
from streamutils.netvlad import NetVLAD, EmbedNet, TripletNet
import stat

from datetime import timedelta
from collections import deque
import dv_processing as dv

import struct


import threading
from queue import Queue, Empty

class RawKPLogger:
    """
    Append-only binary log, fixed-size per frame.
    Stores cropped keypoints (y,x) and later you correct offsets offline.
    """
    MAGIC = b"EKPR"
    VER = 1

    def __init__(self, path: Path, H: int, W: int, off_top: int, off_left: int, top_k: int, D: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.H, self.W = int(H), int(W)
        self.off_top, self.off_left = int(off_top), int(off_left)
        self.top_k, self.D = int(top_k), int(D)

        # Big buffer helps a lot
        self.f = open(self.path, "wb", buffering=1024 * 1024)

        # header: magic(4) ver(u32) H(u16) W(u16) off_top(i16) off_left(i16) K(u16) D(u16)
        hdr = struct.pack(
            "<4sIHHhhHH",
            self.MAGIC, self.VER,
            self.H, self.W,
            self.off_top, self.off_left,
            self.top_k, self.D
        )
        self.f.write(hdr)
        self._frames_since_flush = 0

    @torch.no_grad()
    def write(self, frame_idx: int, t_ref_raw: int, kpts_yx: torch.Tensor, scores: torch.Tensor, desc: torch.Tensor):
        """
        kpts_yx: (N,2) y,x on GPU
        scores:  (N,)  on GPU
        desc:    (N,D) on GPU
        """
        K, D = self.top_k, self.D

        n = int(kpts_yx.shape[0])
        n_valid = min(n, K)

        # Pad to fixed size K so each record is constant-size (fast append)
        k_pad = torch.full((K, 2), -1, device=kpts_yx.device, dtype=torch.int16)
        s_pad = torch.zeros((K,), device=scores.device, dtype=torch.float16)
        d_pad = torch.zeros((K, D), device=desc.device, dtype=torch.float16)

        if n_valid > 0:
            k_pad[:n_valid] = kpts_yx[:n_valid].to(torch.int16)
            s_pad[:n_valid] = scores[:n_valid].to(torch.float16)
            d_pad[:n_valid] = desc[:n_valid].to(torch.float16)

        # move minimal bytes to CPU
        k_np = k_pad.cpu().numpy()
        s_np = s_pad.cpu().numpy()
        d_np = d_pad.cpu().numpy()

        # record header: frame(u32) t_ref(i64) n_valid(u16)
        self.f.write(struct.pack("<IqH", int(frame_idx), int(t_ref_raw), int(n_valid)))
        self.f.write(k_np.tobytes(order="C"))
        self.f.write(s_np.tobytes(order="C"))
        self.f.write(d_np.tobytes(order="C"))

        self._frames_since_flush += 1
        if self._frames_since_flush >= 200:  # tune
            self.f.flush()
            self._frames_since_flush = 0

    def close(self):
        if self.f:
            self.f.flush()
            self.f.close()
            self.f = None

def events_to_polarity_image(H, W, x, y, p, clip_val=5):
    """
    Fast visual: accumulate +/-1 per pixel then map to gray.
    p expected 0/1 (0=neg, 1=pos).
    """
    img = np.zeros((H, W), dtype=np.int16)
    if x.size == 0:
        return np.full((H, W), 127, dtype=np.uint8)

    pos = (p == 1)
    neg = ~pos

    if np.any(pos):
        np.add.at(img, (y[pos], x[pos]), 1)
    if np.any(neg):
        np.add.at(img, (y[neg], x[neg]), -1)

    img = np.clip(img, -clip_val, clip_val)
    disp = ((img.astype(np.float32) + clip_val) / (2.0 * clip_val) * 255.0).astype(np.uint8)
    return disp


class LiveEventPreview:
    """
    Threaded OpenCV preview window. Non-blocking for producer:
    - enqueue() will drop old frames if viewer can't keep up.
    - start() creates window + thread
    - stop() joins thread + destroys window
    """
    def __init__(self, win="DAVIS346 events", scale=1.0, stop_event=None):
        self.win = win
        self.scale = float(scale)
        self.stop_event = stop_event  # optional threading.Event shared with main loop

        self._q = Queue(maxsize=1)     # keep only latest
        self._thread = None
        self._running = threading.Event()

        # for FPS display
        self._last_t = None
        self._fps_ema = None

    def start(self):
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Best-effort window cleanup
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass

    def enqueue(self, H, W, x, y, p, frame_idx, t_ref_raw, t_read_ms):
        """
        Non-blocking producer. Drops frame if viewer is busy.
        Copies are minimal: x/y/p are expected numpy arrays.
        """
        if not self._running.is_set():
            return

        item = (int(H), int(W), x, y, p, int(frame_idx), int(t_ref_raw), float(t_read_ms))

        # Drop old frame if queue is full
        if self._q.full():
            try:
                _ = self._q.get_nowait()
            except Empty:
                pass

        try:
            self._q.put_nowait(item)
        except Exception:
            # If enqueue fails for any reason, just skip (never block main loop)
            pass

    def _loop(self):
        # Some systems benefit from this when running HighGUI in a thread
        try:
            cv2.startWindowThread()
        except Exception:
            pass

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        while self._running.is_set():
            # Try to get latest frame; if none, just idle briefly
            try:
                H, W, x, y, p, frame_idx, t_ref_raw, t_read_ms = self._q.get(timeout=0.05)
            except Empty:
                continue

            now = time.perf_counter()
            if self._last_t is None:
                self._last_t = now
            dt = now - self._last_t
            self._last_t = now
            inst_fps = 1.0 / max(dt, 1e-6)
            self._fps_ema = inst_fps if self._fps_ema is None else (0.9 * self._fps_ema + 0.1 * inst_fps)

            frame = events_to_polarity_image(H, W, x, y, p)

            if self.scale != 1.0:
                frame = cv2.resize(
                    frame,
                    (int(W * self.scale), int(H * self.scale)),
                    interpolation=cv2.INTER_NEAREST
                )

            overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            text1 = f"dt={t_read_ms:.2f}ms  fps~{self._fps_ema:.1f}"
            text2 = f"idx={frame_idx}  events={x.size}  t_ref(us)={t_ref_raw}"
            cv2.putText(overlay, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(self.win, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                if self.stop_event is not None:
                    self.stop_event.set()
                self._running.clear()
                break

        # Cleanup
        try:
            cv2.destroyWindow(self.win)
        except Exception:
            pass

def _apply_davis346_bias_deltas(cap: dv.io.camera.DAVIS, deltas: dict):
    """
    deltas: dict of {dv.io.camera.DAVIS.Davis346BiasCF.<Bias>: int_delta}
    Raw units are whatever dv-processing exposes (docs example uses +1_000_000 steps). :contentReference[oaicite:1]{index=1}
    """
    for bias_enum, delta in deltas.items():
        cur = int(cap.getDavis346BiasCurrent(bias_enum))
        cap.setDavis346BiasCurrent(bias_enum, cur + int(delta))

def stream_event_windows_davis_live(dt_ms: float, on_window=None, bias_deltas=None):
    cap = dv.io.camera.DAVIS()

    # ---- SET ONCE (before enabling event stream) ----
    if bias_deltas:
        # make sure events aren't running yet
        cap.setEventsRunning(False)
        cap.setFramesRunning(False)

        _apply_davis346_bias_deltas(cap, bias_deltas)

        # tiny settle delay (optional, but helps avoid weirdness right at start)
        time.sleep(0.05)

    cap.setEventsRunning(True)
    cap.setFramesRunning(False)

    W, H = cap.getEventResolution()
    slicer = dv.EventStreamSlicer()
    q = deque()

    def cb(events):
        q.append(events)

    slicer.doEveryTimeInterval(timedelta(milliseconds=float(dt_ms)), cb)

    dt_us = int(round(float(dt_ms) * 1000.0))
    frame_idx = 0
    w_end_raw = None

    try:
        while cap.isRunning():
            t0 = time.perf_counter()
            batch = cap.getNextEventBatch()
            t_read_ms = (time.perf_counter() - t0) * 1000.0

            if batch is None:
                time.sleep(0.001)
                continue

            slicer.accept(batch)

            while q:
                ev = q.popleft()

                xy = np.asarray(ev.coordinates())
                if xy.ndim == 2 and xy.shape[0] == 2 and xy.shape[1] != 2:
                    xy = xy.T
                x = xy[:, 0].astype(np.int32, copy=False)
                y = xy[:, 1].astype(np.int32, copy=False)

                t_raw = np.asarray(ev.timestamps()).reshape(-1).astype(np.int64, copy=False)
                p = np.asarray(ev.polarities()).reshape(-1).astype(np.uint8, copy=False)

                if w_end_raw is None:
                    t0_raw = int(t_raw[0]) if t_raw.size else 0
                    w_end_raw = t0_raw + dt_us
                else:
                    w_end_raw += dt_us

                t_ref_raw = w_end_raw

                if on_window is not None:
                    on_window(H, W, x, y, p, frame_idx, t_ref_raw, t_read_ms)

                yield (H, W, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms)
                frame_idx += 1

    finally:
        try:
            cap.setEventsRunning(False)
            cap.setFramesRunning(False)
        except Exception:
            pass
        try:
            cap.close()
        except Exception:
            pass

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
    x, y, p,
    H: int, W: int, device: torch.device
) -> Tuple[torch.Tensor, int]:

    # KEY FIX: keep p as-is, then promote safely before >0
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
    x, y, t, p,
    H, W, t_ref_raw, time_scale, windows_sec: torch.Tensor,
    device
) -> torch.Tensor:
    S = int(windows_sec.numel())
    flat = H * W
    out = torch.zeros((2 * S, H, W), device=device, dtype=torch.float32)

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
        
        return t_kpts, t_desc.to(dtype=torch.float16), t_mask
    
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
    # align dtype/device once
    r_desc = r_desc.to(device=q_desc.device, dtype=q_desc.dtype)
    
    sim = r_desc @ q_desc.T
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

# ---------------------------
# LENS Helpers
# ---------------------------
import torch.nn as nn
class LENS(nn.Module):
    def __init__(self):
        super(LENS, self).__init__()

        # Set the arguments
        input_neurons = 784
        feature_multiplier = 2


        # Change to CPU if selected
        self.device = torch.device('cpu')

        # Layer dict to keep track of layer names and their order
        self.layer_dict = {}
        self.layer_counter = 0

        # Define layer architecture
        self.input = int(input_neurons)
        self.feature = int(self.input*feature_multiplier)
        self.output = int(12824)

        """
        Define trainable layers here
        """
        self.add_layer(
            'feature_layer',
            dims=[self.input, self.feature],
            device=self.device,
            inference=True
        )
        self.add_layer(
            'output_layer',
            dims=[self.feature, self.output],
            device=self.device,
            inference=True
        )

        if not hasattr(self, 'matrix'):
            self.matrix = None

                    # Define convolutional kernel to select the center pixel
        def _init_kernel():
            kernel = torch.zeros(1, 1, 9, 12)
            # Calculate center coordinates for height and width separately
            center_h = 3 // 2
            center_w = 3 // 2
            kernel[0, 0, center_h, center_w] = 1
            return kernel
        
        # Define the Conv2d selection layer
        self.conv = nn.Conv2d(1, 1, kernel_size=(9, 12), stride=(9, 12), padding=0, bias=False).to(self.device)
        self.conv.weight = nn.Parameter(_init_kernel(), requires_grad=False) # Set the kernel weights

        # Define the inferencing forward pass
        self.inference = nn.Sequential(
            self.conv,
            nn.ReLU(),
            nn.Flatten(),
            self.feature_layer.w,
            nn.ReLU(),
            self.output_layer.w,
        )

        # Define the sinabs model, this converts torch model to sinabs model
        from sinabs.from_torch import from_model
        input_shape = (1, 260, 346)
        self.sinabs_model = from_model(
                                self.inference.to(self.device), 
                                input_shape=input_shape,
                                num_timesteps=1000,
                                add_spiking_output=True
        )


    def add_layer(self, name, **kwargs):
        """
        Dynamically add a layer with given name and keyword arguments.
        
        :param name: Name of the layer to be added
        :type name: str
        :param kwargs: Hyperparameters for the layer
        """
        # Check for layer name duplicates
        if name in self.layer_dict:
            raise ValueError(f"Layer with name {name} already exists.")
        
        # Add a new SNNLayer with provided kwargs
        import streamutils.blitnet as bn
        setattr(self, name, bn.SNNLayer(**kwargs))
        
        # Add layer name and index to the layer_dict
        self.layer_dict[name] = self.layer_counter
        self.layer_counter += 1          

    def make_spikes(self, input):
        torch.manual_seed(50)
        img_shape = input.shape

        gen_device = torch.device("cuda") if torch.cuda.is_available() else input.device

        inp = input.to(gen_device)
        r = torch.rand((1000, *inp.shape), device=gen_device, dtype=inp.dtype)
        image = (r < inp).float()

        # (T, 1, H, W)
        image = image.view(1000, img_shape[-2], img_shape[-1]).unsqueeze(1)

        # IMPORTANT: move spikes to the model/device you actually run on (CPU if weights are CPU)
        return image.to(self.device)   # or: return image.cpu()     

    def evaluate(self, spikes):
        """
        Run the inferencing model and calculate the accuracy.

        :param test_loader: Testing data loader
        :param model: Pre-trained network model
        """
        # Run inference for event stream or pre-recorded DVS data
        with torch.no_grad(): 
            spikes= spikes.to(self.device)
            spikes = self.make_spikes(spikes/255)
            spikes = sl.FlattenTime()(spikes)
            # Forward pass
            spikes = self.sinabs_model(spikes.unsqueeze(1))
            output = spikes.sum(dim=0).squeeze()

# ---------------------------
# Sparse Helpers
# ---------------------------
def remove_random_bursts(event_frames, threshold):
    event_frames[event_frames > threshold] = threshold
    return event_frames

def adjust_and_normalize_probabilities(event_data, apply_outlier_correction=True):
    adjusted_probs = np.copy(event_data)

    if apply_outlier_correction:  # Reduce probability for potential outliers
        outlier_threshold = event_data.mean() + 2 * event_data.std()
        adjusted_probs[adjusted_probs > outlier_threshold] = 0.01

    total_prob = adjusted_probs.sum()
    normalized_probs = adjusted_probs / total_prob
    return normalized_probs

def get_random_pixels(num_pixels, im_width, im_height, local_suppression_radius, prob_to_draw_from=None):
    """
    Generate a list of random pixels within an image.

    Args:
        num_pixels (int): The number of random pixels to generate.
        im_width (int): The width of the image.
        im_height (int): The height of the image.
        local_suppression_radius (float): The radius for local suppression.
        prob_to_draw_from (ndarray, optional): The probability distribution to draw from. Defaults to None.

    Returns:
        list: A list of random pixels, each represented as a tuple (y, x).

    Raises:
        ValueError: If a new random pixel cannot be found after 100 iterations.
    """
    random_pixels = []
    num_subsequent_rejections = 0
    with tqdm(total=num_pixels, desc="Pick random pixels") as pbar:
        while len(random_pixels) < num_pixels:
            random_idx_flat = np.random.choice(
                np.arange(0, im_height * im_width),
                p=prob_to_draw_from.reshape(-1) if prob_to_draw_from is not None else None,
            )
            random_pixel = np.unravel_index(random_idx_flat, (im_height, im_width))
            if len(random_pixels) == 0 or np.all(np.linalg.norm(np.array(random_pixels) - np.array(random_pixel), axis=1) > local_suppression_radius):
                random_pixels.append(random_pixel)
                num_subsequent_rejections = 0
                pbar.update(1)
            else:
                num_subsequent_rejections = num_subsequent_rejections + 1
                if num_subsequent_rejections > 100:
                    raise ValueError("Could not find new random pixel after 100 iterations")

    # check that number of unique elements equals the number of requested pixels
    assert len(list(set(random_pixels))) == num_pixels

    return random_pixels

def get_distance_matrix(ref_traverse: np.ndarray, qry_traverse: np.ndarray, metric="cosine", device='cuda'):
    dev = device

    a = torch.from_numpy(ref_traverse.reshape(ref_traverse.shape[0], -1).astype(np.float32)).unsqueeze(0).to(dev)
    b = qry_traverse.reshape(qry_traverse.shape[0], -1).float().unsqueeze(0).to(dev)
    if metric == "cityblock":
        torch_dist = torch.cdist(a, b, 1)[0]
    elif metric == "euclidean":
        torch_dist = torch.cdist(a, b, 2)[0]
    elif metric == "cosine":
        def cosine_distance_torch(x1, x2=None, eps=1e-8):
            x2 = x1 if x2 is None else x2
            w1 = x1.norm(p=2, dim=1, keepdim=True)
            w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
            return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

        torch_dist = cosine_distance_torch(a.squeeze(0), b.squeeze(0))
    else:
        raise ValueError("Distance not supported")

    if device == torch.device("mps"):
        torch_dist = torch_dist.to(device)

    return torch_dist

# ---------------------------
# EventVLAD Helpers
# ---------------------------
class DnCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dep=20, num_filters=64, slope=0.2):
        '''
        Reference:
        K. Zhang, W. Zuo, Y. Chen, D. Meng and L. Zhang, "Beyond a Gaussian Denoiser: Residual
        Learning of Deep CNN for Image Denoising," TIP, 2017.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            dep (int): depth of the network, Default 20
            num_filters (int): number of filters in each layer, Default 64
        '''
        super(DnCNN, self).__init__()
        self.conv1 = conv3x3(in_channels, num_filters, bias=True)
        self.relu = nn.LeakyReLU(slope, inplace=True)
        mid_layer = []
        for ii in range(1, dep-1):
            mid_layer.append(conv3x3(num_filters, num_filters, bias=True))
            mid_layer.append(nn.LeakyReLU(slope, inplace=True))
        self.mid_layer = nn.Sequential(*mid_layer)
        self.conv_last = conv3x3(num_filters, out_channels, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mid_layer(x)
        out = self.conv_last(x)

        return out

def weight_init_kaiming(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if not m.bias is None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return net

class EventDenoiser(nn.Module):
    def __init__(self, input_images, dep_S=5, dep_U=4, slope=0.2):
        super(EventDenoiser, self).__init__()
        config = {'num_bins' : 3}
        self.ReconNet = RecurrUNet(num_bins = 3, in_channels = 1, out_channels = 1, depth=dep_U, slope=slope)
        self.ErrorNet = DnCNN(in_channels = 3, out_channels = 1, dep=dep_S, num_filters=64, slope=slope)

    def forward(self, x):
        img_estim = self.ReconNet(x)
        err_estim = self.ErrorNet(x)
        evterr = torch.cat((img_estim,err_estim),dim=1)
        return evterr

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

class BaseE2VID(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        try:
            self.skip_type = str(config['skip_type'])
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.num_encoders = int(config['num_encoders'])
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.norm = str(config['norm'])
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])
        except KeyError:
            self.use_upsample_conv = True

class RecurrUNet(BaseE2VID):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, num_bins = 3, in_channels=1, out_channels=2, depth=4, slope=0.2):
        self.output_channels = out_channels
        self.num_encoders = depth
        self.base_num_channels = out_channels
        self.num_residual_blocks = depth
        self.in_channels = in_channels
        self.num_bins = num_bins  # number of bins in the voxel grid event tensor
        config = {}
        super(RecurrUNet, self).__init__(config)

        try:
            self.recurrent_block_type = str(config['recurrent_block_type'])
        except KeyError:
            self.recurrent_block_type = 'convgru'  # or 'convlstm'

#        self.unetrecurrent = UNet(num_input_channels=self.in_channels,
#                                           num_output_channels=self.output_channels,
#                                           skip_type='sum',
#                                           activation='sigmoid',
#                                           num_encoders=self.num_encoders,
#                                           base_num_channels=self.base_num_channels,
#                                           num_residual_blocks=self.num_residual_blocks,
#                                           norm=self.norm,
#                                           use_upsample_conv=self.use_upsample_conv)
        self.unetrecurrent = UNetRecurrent(num_input_channels=self.in_channels,
                                           num_output_channels=self.output_channels,
                                           skip_type='sum',
                                           recurrent_block_type=self.recurrent_block_type,
                                           activation='sigmoid',
                                           num_encoders=self.num_encoders,
                                           base_num_channels=self.base_num_channels,
                                           num_residual_blocks=self.num_residual_blocks,
                                           norm=self.norm,
                                           use_upsample_conv=self.use_upsample_conv)

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :param prev_states: previous ConvLSTM state for each encoder module
        :return: reconstructed image, taking values in [0,1].
        """
#        img_pred = self.unetrecurrent.forward(event_tensor)
        states = None
        num_bins = event_tensor.shape[1]
        for nth in range(num_bins):
            eventimg = event_tensor[:,nth,]
            eventimg = eventimg[:,np.newaxis,]
            img_pred, states = self.unetrecurrent.forward(eventimg, states)
        return img_pred

def skip_sum(x1, x2):
    return torch.add(x1,x2)

class BaseUNet(nn.Module):
    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(BaseUNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum
        self.activation = activation
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert(self.num_input_channels > 0)
        assert(self.num_output_channels > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)

class UNetRecurrent(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, num_input_channels, num_output_channels=1, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=4, base_num_channels=32,
                 num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(UNetRecurrent, self).__init__(num_input_channels, num_output_channels, skip_type, activation,
                                            num_encoders, base_num_channels, num_residual_blocks, norm,
                                            use_upsample_conv)

        self.head = ConvLayer(self.num_input_channels, self.base_num_channels,
                              kernel_size=5, stride=1, padding=2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(input_size, output_size,
                                                    kernel_size=5, stride=2, padding=2,
                                                    recurrent_block_type=recurrent_block_type,
                                                    norm=self.norm))

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, prev_states):
        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x)
        head = x

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, prev_states[i])
            blocks.append(x)
            states.append(state)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
#        img = self.activation(self.pred(self.apply_skip_connectiomn(x, head)))
        img = self.pred(self.apply_skip_connection(x, head))

        return img, states
    
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        nn.init.xavier_uniform_(self.conv2d.weight)
        
        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        nn.init.xavier_uniform_(self.conv2d.weight)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None):
        super(RecurrentConvLayer, self).__init__()

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)

    def forward(self, x, prev_state):
        x = self.conv(x)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class DownsampleRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0, activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation, 'relu')

        assert(recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state

def build_model(model_type: str, dep_u: int, dep_s: int, slope: float) -> torch.nn.Module:
    mt = model_type.lower()
    if mt in ["event_denoiser", "denoiser", "default"]:
        return EventDenoiser(3, slope=slope, dep_U=dep_u, dep_S=dep_s)
    raise ValueError(f"Unknown model_type: {model_type}")

def clean_state_dict(sd):
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def load_checkpoint_into_model(model, ckpt_path, use_gpu=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    sd = clean_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] Missing keys: {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")
    if use_gpu and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    model.eval()
    return model

def make_divisible(img, mult):
    if mult <= 1:
        return img
    H, W = img.shape[:2]
    return img[: H - (H % mult) if H % mult else H,
               : W - (W % mult) if W % mult else W]

def prep_input_triplet(i0: np.ndarray, i1: np.ndarray, i2: np.ndarray,
                       size: int, dep_u: int, rotate180: bool) -> torch.Tensor:
    """Prepare a 1x3xHxW tensor from three grayscale [0,1] images."""
    m = 2 ** dep_u if dep_u > 0 else 1
    i0 = make_divisible(i0, m)
    i1 = make_divisible(i1, m)
    i2 = make_divisible(i2, m)

    if size and size > 0:
        i0 = cv2.resize(i0, (size, size), interpolation=cv2.INTER_AREA)
        i1 = cv2.resize(i1, (size, size), interpolation=cv2.INTER_AREA)
        i2 = cv2.resize(i2, (size, size), interpolation=cv2.INTER_AREA)

    if rotate180:
        i0 = cv2.rotate(i0, cv2.ROTATE_180)
        i1 = cv2.rotate(i1, cv2.ROTATE_180)
        i2 = cv2.rotate(i2, cv2.ROTATE_180)

    t0 = torch.from_numpy(i0[None, ...])  # (1,H,W)
    t1 = torch.from_numpy(i1[None, ...])
    t2 = torch.from_numpy(i2[None, ...])
    x = torch.cat([t0, t1, t2], dim=0)[None, ...].contiguous().float()  # (1,3,H,W)
    return x

def tensor_to_uint8(img_t):
    """Accepts (1,1,H,W) or (1,C,H,W) or (H,W), returns uint8 HxW image."""
    if torch.is_tensor(img_t):
        img = img_t.detach().cpu().numpy()
    else:
        img = np.asarray(img_t)
    if img.ndim == 4:
        img = img[:, 0, ...]
    if img.ndim == 3:
        img = img[0]
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def stream_vlad_denoise(queries, model):
    x = prep_input_triplet(queries[0], queries[1], queries[2], size=256, dep_u=5, rotate180=False)
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    with torch.no_grad():
        out = model(x)
    return tensor_to_uint8(out)

def _device(dev=None):
    if isinstance(dev, torch.device): return dev
    if isinstance(dev, str): return torch.device(dev)
    if torch.cuda.is_available(): return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _safe_torch_load(path):
    # Prefer safe loading when available; fall back quietly on older torch.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _strip_prefix(k, prefix):
    return k[len(prefix):] if k.startswith(prefix) else k

def _remap_state_for_triplet(state):
    """
    Map various checkpoint layouts to TripletNet(embed_net(base_model, net_vlad)) EXACT names:
      - VGG:   embed_net.base_model.<layer>
      - NetVLAD: embed_net.net_vlad.<param>
    """
    if "state_dict" in state: state = state["state_dict"]
    if "model_state_dict" in state: state = state["model_state_dict"]

    fixed = {}
    for k, v in state.items():
        # strip common wrappers
        for pref in ("module.", "model.", "net.", "triplet.", "tripletnet.", "triplet_model.", "embed.", "embed_net."):
            k = _strip_prefix(k, pref)

        # normalize obvious roots
        if k.startswith("base_model."):
            k = "embed_net.base_model." + k[len("base_model."):]
        elif k.startswith("net_vlad."):
            k = "embed_net.net_vlad." + k[len("net_vlad."):]
        elif k.startswith("encoder."):
            # many checkpoints store VGG as "encoder.*"
            k = "embed_net.base_model." + k[len("encoder."):]
        elif k.startswith("vlad.") or k.startswith("netvlad."):
            k = "embed_net.net_vlad." + k.split(".", 1)[1]
        elif k.startswith("pool."):
            # some store NetVLAD as "pool.*"
            k = "embed_net.net_vlad." + k[len("pool."):]
        elif k.startswith("embed_net.base_model.pool."):
            # rare mis-save of vlad under base_model.pool.*
            k = "embed_net.net_vlad." + k[len("embed_net.base_model.pool."):]
        else:
            # bare VGG layer names (convX_Y, reluX_Y, poolX, fc6/fc7/fc8)
            if k.startswith(("conv1_","conv2_","conv3_","conv4_","conv5_","relu","pool","fc6","fc7","fc8")):
                k = "embed_net.base_model." + k
            # bare NetVLAD param names
            elif k.startswith(("centroids", "conv.weight", "conv.bias", "lastfc.weight", "lastfc.bias")):
                k = "embed_net.net_vlad." + k

        fixed[k] = v
    return fixed

def _infer_vlad_dims(mapped_state):
    """
    Infer (K, D) for NetVLAD from the mapped state.
    """
    if "embed_net.net_vlad.centroids" in mapped_state:
        w = mapped_state["embed_net.net_vlad.centroids"]  # [K, D]
        return int(w.shape[0]), int(w.shape[1])
    if "embed_net.net_vlad.conv.weight" in mapped_state:
        w = mapped_state["embed_net.net_vlad.conv.weight"]  # [K, D, 1, 1]
        return int(w.shape[0]), int(w.shape[1])
    raise RuntimeError("Could not infer NetVLAD (K, D): no centroids or conv.weight in checkpoint.")

def _ensure_required_vlad_params(mapped):
    """
    Some checkpoints omit NetVLAD conv.bias. If the model expects it,
    synthesize a zero bias so strict=True can succeed.
    """
    need_bias_key = "embed_net.net_vlad.conv.bias"
    if need_bias_key not in mapped:
        # infer K (num_clusters)
        if "embed_net.net_vlad.conv.weight" in mapped:
            K = int(mapped["embed_net.net_vlad.conv.weight"].shape[0])
            dtype = mapped["embed_net.net_vlad.conv.weight"].dtype
        elif "embed_net.net_vlad.centroids" in mapped:
            K = int(mapped["embed_net.net_vlad.centroids"].shape[0])
            dtype = mapped["embed_net.net_vlad.centroids"].dtype
        else:
            raise RuntimeError("Cannot infer K to synthesize conv.bias.")

        mapped[need_bias_key] = torch.zeros(K, dtype=dtype)  # CPU is fine
    return mapped

def isfile(path):
    """Test whether a path is a regular file"""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(st.st_mode)

def build_eventvlad_model_from_tar(weights_path: str, num_clusters: int = 64, device=None):
    """
    Build TripletNet(EmbedNet(Imagenet_vgg, NetVLAD)) and load weights from .tar strictly.
    NOTE: num_clusters is ignored if the checkpoint indicates a different K; we match the checkpoint.
    """
    assert isfile(weights_path), f"Missing weights file: {weights_path}"
    dev = _device(device)

    # Base (their MatConvNet-style VGG16 -> fc8:1000)
    base = Imagenet_vgg(weights_path=None)  # we load from the .tar, not a separate .pth

    # Load and map checkpoint FIRST so we can infer dims
    raw = _safe_torch_load(weights_path)
    mapped = _remap_state_for_triplet(raw)
    K, D = _infer_vlad_dims(mapped)  # e.g., K=64, D=1000
    mapped = _ensure_required_vlad_params(mapped)

    # Build NetVLAD to EXACT dims from checkpoint
    vlad = NetVLAD(num_clusters=K, dim=D, normalize_input=True)

    embed = EmbedNet(base, vlad)
    model = TripletNet(embed).to(dev).eval()

    # Pre-validate to ensure strict=True will pass (no surprises)
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(mapped.keys())
    missing = sorted(model_keys - mapped_keys)
    unexpected = sorted(mapped_keys - model_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Key mismatch after mapping.\n"
            f"Missing ({len(missing)}): {missing[:12]}{' ...' if len(missing)>12 else ''}\n"
            f"Unexpected ({len(unexpected)}): {unexpected[:12]}{' ...' if len(unexpected)>12 else ''}"
        )

    model.load_state_dict(mapped, strict=True)
    return model


VGG_MEAN_RGB = np.array([122.7449417, 114.9440994, 101.6417770], dtype=np.float32)

def preprocess_like_eventvgg(query_np, rgb_input=True, mean_rgb=VGG_MEAN_RGB):
    """
    Mimics _EventVGGPreprocess._load() for ndarray inputs.

    Notes:
    - If rgb_input=True, this assumes any 3-ch input is BGR and converts to RGB
      (matching your dataset code's behavior).
    - Does NOT divide by 255.
    - Output: torch.FloatTensor [3, 224, 224]
    """
    x = query_np
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    # If CHW, convert to HWC (best-effort heuristic)
    if x.ndim == 3 and x.shape[0] in (1, 2, 3, 4) and x.shape[-1] not in (1, 2, 3, 4):
        x = np.transpose(x, (1, 2, 0))  # CHW -> HWC

    if x.ndim == 2:
        x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)  # -> HWC, 3ch
    elif x.ndim == 3:
        if x.shape[2] == 1:
            x = np.repeat(x, 3, axis=2)
        elif x.shape[2] == 2:
            # Your original pipeline likely never had 2ch here (cv2.cvtColor would choke).
            # If it does happen, pad a 3rd channel with zeros to keep the VGG mean logic valid.
            z = np.zeros_like(x[..., :1])
            x = np.concatenate([x, z], axis=2)
        elif x.shape[2] == 4:
            x = cv2.cvtColor(x, cv2.COLOR_BGRA2BGR)
            if rgb_input:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        elif x.shape[2] == 3:
            if rgb_input:
                # IMPORTANT: matches your Dataset behavior (treat ndarray as BGR)
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count: {x.shape[2]}")
    else:
        raise ValueError(f"Unsupported query shape: {x.shape}")

    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)

    x = x.astype(np.float32)
    x -= mean_rgb  # RGB means
    x = np.transpose(x, (2, 0, 1))  # CHW

    return torch.from_numpy(x)  # float32


@torch.inference_mode()
def extract_eventvlad_features(model, query, device='cuda'):
    dev = _device(device)
    model = model.to(dev).eval()

    q = preprocess_like_eventvgg(query, rgb_input=True).to(dev)  # [3,224,224]
    out = model.feature_extract(q.unsqueeze(0))                  # [1,D] (or sometimes [D])

    if out.dim() == 1:
        out = out.unsqueeze(0)

    return out.detach().cpu().to(torch.float32).numpy()
