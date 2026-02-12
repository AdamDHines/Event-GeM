import argparse
import asyncio
import base64
import json
import threading
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import h5py
import numpy as np
import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    hdf5: Path
    host: str = "127.0.0.1"
    port: int = 8000

    dt_ms: float = 50.0
    target_fps: float = 20.0
    realtime: bool = True

    chunk_size: int = 250_000
    time_scale: float = 1e-9  # brisbane_event epoch ns -> seconds

    start_time: Optional[float] = None
    max_frames: Optional[int] = None

    height: int = 0
    width: int = 0
    infer_full_scan: bool = False

    jpeg_quality: int = 80

    # UI: downscale sent images to reduce bandwidth/CPU decode
    viz_scale: float = 0.6

    # MCTS time windows (ms) -> 5 windows => 10 channels (pos+neg)
    mcts_windows_ms: Tuple[int, ...] = (10, 20, 30, 40, 50)

    # Use thread pool to build polarity + MCTS in parallel
    parallel_build: bool = True


CONFIG: Optional[Config] = None


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
# Time helpers: stay integer in raw units
# ---------------------------

def sec_to_raw(t_sec: float, time_scale: float) -> int:
    return int(round(float(t_sec) / float(time_scale)))


def raw_to_sec(t_raw: int, time_scale: float) -> float:
    return float(t_raw) * float(time_scale)


# ---------------------------
# Window streaming (mask-based, robust)
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

    print(f"[INFO] dt_ms={dt_ms} => dt_raw={dt_raw} (time_scale={time_scale})")
    print(f"[INFO] file t range raw: [{t0_raw}, {tN_raw}]  sec: [{raw_to_sec(t0_raw,time_scale):.6f}, {raw_to_sec(tN_raw,time_scale):.6f}]")
    print(f"[INFO] start at raw={w_start_raw} sec={raw_to_sec(w_start_raw,time_scale):.6f}")

    while w_start_raw < tN_raw and (max_frames is None or frame_idx < max_frames):
        w_end_raw = w_start_raw + dt_raw

        # Fill buffer until we have timestamps beyond window end (or EOF)
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

        # Drop old events
        if t_buf.size:
            keep = t_buf >= w_start_raw
            x_buf, y_buf, t_buf, p_buf = x_buf[keep], y_buf[keep], t_buf[keep], p_buf[keep]
            t_buf_max = int(np.max(t_buf)) if t_buf.size else -1

        # Select window events
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

        yield (
            raw_to_sec(w_start_raw, time_scale),
            raw_to_sec(w_end_raw, time_scale),
            w_end_raw,              # raw ref time for MCTS
            x_win, y_win, t_win_raw, p_win,
            frame_idx,
        )

        w_start_raw = w_end_raw
        frame_idx += 1

        if read_idx >= N and t_buf.size == 0 and t_buf_max < w_start_raw:
            break

    f.close()


# ---------------------------
# Representation builders
# ---------------------------

def polarity_hist_rgb(x: np.ndarray, y: np.ndarray, p: np.ndarray, height: int, width: int) -> np.ndarray:
    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)

    if x.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return np.zeros((height, width, 3), dtype=np.uint8)

    xv = x[valid]
    yv = y[valid]
    pv = p[valid]

    pos_mask = pv > 0
    neg_mask = ~pos_mask

    if np.any(pos_mask):
        np.add.at(pos, (yv[pos_mask], xv[pos_mask]), 1.0)
    if np.any(neg_mask):
        np.add.at(neg, (yv[neg_mask], xv[neg_mask]), 1.0)

    m = max(np.percentile(pos, 99.5), np.percentile(neg, 99.5), 1.0)
    pos_n = np.clip(pos / m, 0.0, 1.0)
    neg_n = np.clip(neg / m, 0.0, 1.0)

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 0] = pos_n  # R
    rgb[..., 2] = neg_n  # B
    return (rgb * 255.0).astype(np.uint8)


def encode_jpeg_b64(img_u8: np.ndarray, quality: int = 80) -> str:
    if img_u8.ndim == 2:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", img_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------
# MCTS (NumPy) + visualization mosaic
# ---------------------------

def mcts_numpy(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    height: int,
    width: int,
    t_ref: float,
    time_windows: np.ndarray,  # seconds
) -> np.ndarray:
    num_scales = len(time_windows)
    C = 2 * num_scales
    mcts = np.zeros((C, height, width), dtype=np.float32)

    if x.size == 0:
        return mcts

    x = x.astype(np.int64, copy=False)
    y = y.astype(np.int64, copy=False)
    t = t.astype(np.float64, copy=False)
    p = p.astype(np.int8, copy=False)

    valid_xy = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid_xy], y[valid_xy], t[valid_xy], p[valid_xy]
    if x.size == 0:
        return mcts

    for pol_idx, pol_sign in enumerate((1, -1)):
        mask_pol = (p > 0) if pol_sign > 0 else (p <= 0)
        if not np.any(mask_pol):
            continue

        x_p, y_p, t_p = x[mask_pol], y[mask_pol], t[mask_pol]
        idx_flat_all = y_p * width + x_p

        for s_idx, DeltaT in enumerate(time_windows):
            DeltaT = float(DeltaT)
            window_start = t_ref - DeltaT

            in_win = t_p >= window_start
            if not np.any(in_win):
                continue

            idx_flat = idx_flat_all[in_win]
            t_valid = t_p[in_win]

            idx_rev = idx_flat[::-1]
            t_rev = t_valid[::-1]

            unique_pix, first_idx_rev = np.unique(idx_rev, return_index=True)
            last_idx = len(idx_rev) - 1 - first_idx_rev
            t_last = t_valid[last_idx]

            y_last = (unique_pix // width).astype(np.int64)
            x_last = (unique_pix % width).astype(np.int64)

            dt = t_ref - t_last
            dt = np.clip(dt, 0.0, DeltaT)
            values = np.exp(-dt / DeltaT).astype(np.float32)

            ch = pol_idx * num_scales + s_idx
            mcts[ch, y_last, x_last] = values

    return mcts


def mcts_to_rgb_mosaic(
    mcts: np.ndarray,
    windows_ms: Tuple[int, ...],
    add_labels: bool = True,
) -> np.ndarray:
    """
    Build a 1xK mosaic (K=number of time windows).
    Each tile: pos channel -> red, neg channel -> blue for that window.
    """
    num_scales = len(windows_ms)
    H, W = mcts.shape[1], mcts.shape[2]

    tiles: List[np.ndarray] = []
    for i, ms in enumerate(windows_ms):
        pos = mcts[i]
        neg = mcts[num_scales + i]

        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[..., 0] = pos
        rgb[..., 2] = neg
        tile = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        if add_labels:
            cv2.putText(
                tile, f"{ms}ms",
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        tiles.append(tile)

    mosaic = np.concatenate(tiles, axis=1)  # H x (W*K) x 3
    return mosaic


def resize_for_viz(img: np.ndarray, scale: float) -> np.ndarray:
    if scale is None or abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


# ---------------------------
# Web UI (fix blinking: reuse Image objects, don't resize every frame, hold last on 0 events)
# ---------------------------

INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Event Stream Demo</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 16px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
    .panel { border: 1px solid #ddd; border-radius: 12px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
    .title { font-weight: 700; margin-bottom: 8px; }
    canvas { width: 100%; height: auto; background: #111; border-radius: 8px; }
    .stats { margin-top: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; white-space: pre; }
  </style>
</head>
<body>
  <h2>HDF5 Event Stream → Browser (target 20 FPS)</h2>
  <div class="grid">
    <div class="panel">
      <div class="title">Window 1 — Polarity histogram (pos=red, neg=blue)</div>
      <canvas id="c1"></canvas>
    </div>
    <div class="panel">
      <div class="title">Window 2 — MCTS mosaic (10 channels → 5 windows)</div>
      <canvas id="c2"></canvas>
    </div>
  </div>

  <div class="panel" style="margin-top:16px;">
    <div class="title">Stats</div>
    <div id="stats" class="stats">Connecting...</div>
  </div>

<script>
  const c1 = document.getElementById("c1");
  const c2 = document.getElementById("c2");
  const s  = document.getElementById("stats");
  const ctx1 = c1.getContext("2d");
  const ctx2 = c2.getContext("2d");

  // Reuse Image objects (avoid flicker + GC churn)
  const img1 = new Image();
  const img2 = new Image();

  let lastT = performance.now();
  let fpsEMA = 0;

  function drawB64(canvas, ctx, imgObj, b64, w, h) {
    if (!b64) return;

    // Only resize if dimensions changed (resizing clears the canvas)
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    imgObj.onload = () => { ctx.drawImage(imgObj, 0, 0, w, h); };
    imgObj.src = "data:image/jpeg;base64," + b64;
  }

  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { s.textContent = "Connected."; };
  ws.onclose = () => { s.textContent = "Disconnected."; };
  ws.onerror = () => { s.textContent = "WebSocket error."; };

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    const now = performance.now();
    const instFPS = 1000.0 / Math.max(1.0, (now - lastT));
    lastT = now;
    fpsEMA = fpsEMA ? (0.9 * fpsEMA + 0.1 * instFPS) : instFPS;

    // Hold last visuals on empty windows to avoid "blink black".
    // Stats will still show events=0 when that happens.
    if (msg.n_events > 0) {
      drawB64(c1, ctx1, img1, msg.pol_jpg, msg.pol_w, msg.pol_h);
      drawB64(c2, ctx2, img2, msg.mcts_jpg, msg.mcts_w, msg.mcts_h);
    }

    s.textContent =
`frame: ${msg.frame_idx}
events: ${msg.n_events}   valid: ${msg.n_valid}
window: [${msg.window_start_s.toFixed(6)}, ${msg.window_end_s.toFixed(6)}]
pol:  ${msg.pol_h}x${msg.pol_w}   b64=${msg.len_pol}
mcts: ${msg.mcts_h}x${msg.mcts_w} b64=${msg.len_mcts}
backend_ms: pol=${msg.t_pol_ms.toFixed(2)} mcts=${msg.t_mcts_ms.toFixed(2)} enc=${msg.t_encode_ms.toFixed(2)}
fps(client_ema): ${fpsEMA.toFixed(1)}  target: ${msg.target_fps}
`;
  };
</script>
</body>
</html>
"""


# ---------------------------
# FrameHub
# ---------------------------

class FrameHub:
    def __init__(self, cfg: Config, H: int, W: int):
        self.cfg = cfg
        self.H = H
        self.W = W
        self.subscribers: set[asyncio.Queue] = set()
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._producer_started = False

        self._pool = ThreadPoolExecutor(max_workers=2) if cfg.parallel_build else None
        self._mcts_windows_sec = np.array(cfg.mcts_windows_ms, dtype=np.float64) * 1e-3

    async def start(self):
        self._loop = asyncio.get_running_loop()
        print("[HUB] start(): event loop captured")

    def stop(self):
        self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self.subscribers.add(q)
        if not self._producer_started:
            self._producer_started = True
            print("[HUB] First subscriber connected → starting producer thread")
            threading.Thread(target=self._producer_loop, daemon=True).start()
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self.subscribers.discard(q)

    def _fanout(self, payload: dict):
        for q in list(self.subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    # ---- build helpers ----

    def _build_polarity_packet(self, x, y, p) -> Tuple[str, int, int, float]:
        t0 = time.perf_counter()
        pol = polarity_hist_rgb(x, y, p, self.H, self.W)
        pol = resize_for_viz(pol, self.cfg.viz_scale)
        pol_jpg = encode_jpeg_b64(pol, quality=self.cfg.jpeg_quality)
        t1 = time.perf_counter()
        return pol_jpg, pol.shape[1], pol.shape[0], (t1 - t0) * 1000.0  # b64, w, h, ms

    def _build_mcts_packet(self, x, y, t_raw, p, t_ref_raw) -> Tuple[str, int, int, float]:
        t0 = time.perf_counter()

        # Convert raw timestamps to relative seconds (stable), t_ref=0
        # t_rel <= 0 for events before t_ref
        t_rel = (t_raw.astype(np.int64, copy=False) - int(t_ref_raw)) * float(self.cfg.time_scale)
        t_ref = 0.0

        mcts = mcts_numpy(
            x=x,
            y=y,
            t=t_rel,
            p=p,
            height=self.H,
            width=self.W,
            t_ref=t_ref,
            time_windows=self._mcts_windows_sec,
        )

        mosaic = mcts_to_rgb_mosaic(mcts, self.cfg.mcts_windows_ms, add_labels=True)
        mosaic = resize_for_viz(mosaic, self.cfg.viz_scale)

        mcts_jpg = encode_jpeg_b64(mosaic, quality=self.cfg.jpeg_quality)
        t1 = time.perf_counter()
        return mcts_jpg, mosaic.shape[1], mosaic.shape[0], (t1 - t0) * 1000.0

    def _producer_loop(self):
        assert self._loop is not None, "Hub event loop not set (startup failed)"

        try:
            target_dt = 1.0 / max(1e-6, self.cfg.target_fps)

            while not self._stop.is_set():
                print("[PROD] Starting stream:", self.cfg.hdf5)

                for (w0, w1, t_ref_raw, x, y, t_raw, p, frame_idx) in stream_event_windows_raw(
                    self.cfg.hdf5,
                    dt_ms=self.cfg.dt_ms,
                    chunk_size=self.cfg.chunk_size,
                    time_scale=self.cfg.time_scale,
                    start_time_sec=self.cfg.start_time,
                    max_frames=self.cfg.max_frames,
                ):
                    if self._stop.is_set():
                        break

                    tick = time.perf_counter()

                    valid = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
                    n_valid = int(valid.sum())

                    # Build in parallel (polarity + mcts) if enabled
                    if self._pool is not None:
                        f_pol = self._pool.submit(self._build_polarity_packet, x, y, p)
                        f_mcts = self._pool.submit(self._build_mcts_packet, x, y, t_raw, p, t_ref_raw)

                        pol_jpg, pol_w, pol_h, t_pol_ms = f_pol.result()
                        mcts_jpg, mcts_w, mcts_h, t_mcts_ms = f_mcts.result()
                    else:
                        pol_jpg, pol_w, pol_h, t_pol_ms = self._build_polarity_packet(x, y, p)
                        mcts_jpg, mcts_w, mcts_h, t_mcts_ms = self._build_mcts_packet(x, y, t_raw, p, t_ref_raw)

                    t_encode_ms = (time.perf_counter() - tick) * 1000.0 - (t_pol_ms + t_mcts_ms)

                    payload = {
                        "frame_idx": int(frame_idx),
                        "window_start_s": float(w0),
                        "window_end_s": float(w1),
                        "n_events": int(x.size),
                        "n_valid": int(n_valid),

                        "pol_jpg": pol_jpg,
                        "pol_w": int(pol_w),
                        "pol_h": int(pol_h),
                        "len_pol": len(pol_jpg),

                        "mcts_jpg": mcts_jpg,
                        "mcts_w": int(mcts_w),
                        "mcts_h": int(mcts_h),
                        "len_mcts": len(mcts_jpg),

                        "t_pol_ms": float(t_pol_ms),
                        "t_mcts_ms": float(t_mcts_ms),
                        "t_encode_ms": float(max(0.0, t_encode_ms)),
                        "target_fps": float(self.cfg.target_fps),
                    }

                    self._loop.call_soon_threadsafe(self._fanout, payload)

                    if self.cfg.realtime:
                        elapsed = time.perf_counter() - tick
                        time.sleep(max(0.0, target_dt - elapsed))

                if self._stop.is_set():
                    break
                print("[PROD] Rewinding file and looping again...")

        except Exception:
            import traceback
            print("[PROD] FATAL producer exception:")
            traceback.print_exc()


# ---------------------------
# FastAPI app
# ---------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global CONFIG
    if CONFIG is None:
        raise RuntimeError("CONFIG not set. Run via `python stream.py --hdf5 ...`.")

    print("[STARTUP] opening:", CONFIG.hdf5)
    with h5py.File(CONFIG.hdf5, "r") as f:
        x_dset, y_dset, t_dset, p_dset = find_event_datasets(f)
        print("[DBG] x:", x_dset.name, x_dset.dtype, x_dset.shape)
        print("[DBG] y:", y_dset.name, y_dset.dtype, y_dset.shape)
        print("[DBG] t:", t_dset.name, t_dset.dtype, t_dset.shape)
        print("[DBG] p:", p_dset.name, p_dset.dtype, p_dset.shape)
        print("[DBG] t0,tN raw:", int(t_dset[0]), int(t_dset[-1]))

        if CONFIG.height > 0 and CONFIG.width > 0:
            H, W = CONFIG.height, CONFIG.width
        else:
            H, W = infer_resolution(x_dset, y_dset, CONFIG.chunk_size, CONFIG.infer_full_scan)

    print(f"[STARTUP] Using HxW = {H}x{W} (viz_scale={CONFIG.viz_scale})")
    app.state.hub = FrameHub(CONFIG, H, W)
    await app.state.hub.start()
    yield
    app.state.hub.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub = getattr(app.state, "hub", None)
    if hub is None:
        await ws.send_text(json.dumps({"error": "hub not initialized"}))
        await ws.close()
        return

    q = hub.subscribe()
    try:
        while True:
            msg = await q.get()
            await ws.send_text(json.dumps(msg))
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(q)


# ---------------------------
# main
# ---------------------------

def main():
    global CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=str, required=True)

    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)

    ap.add_argument("--dt-ms", type=float, default=50.0)
    ap.add_argument("--target-fps", type=float, default=20.0)

    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--no-realtime", dest="realtime", action="store_false")
    ap.set_defaults(realtime=True)

    ap.add_argument("--chunk-size", type=int, default=250_000)
    ap.add_argument("--time-scale", type=float, default=1e-9)
    ap.add_argument("--start-time", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)

    ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--infer-full-scan", action="store_true")

    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--viz-scale", type=float, default=0.6)

    ap.add_argument("--no-parallel", dest="parallel_build", action="store_false")
    ap.set_defaults(parallel_build=True)

    args = ap.parse_args()

    CONFIG = Config(
        hdf5=Path(args.hdf5),
        host=args.host,
        port=args.port,
        dt_ms=args.dt_ms,
        target_fps=args.target_fps,
        realtime=args.realtime,
        chunk_size=args.chunk_size,
        time_scale=args.time_scale,
        start_time=args.start_time,
        max_frames=args.max_frames,
        height=args.height,
        width=args.width,
        infer_full_scan=args.infer_full_scan,
        jpeg_quality=args.jpeg_quality,
        viz_scale=args.viz_scale,
        parallel_build=args.parallel_build,
    )

    import uvicorn
    uvicorn.run(app, host=CONFIG.host, port=CONFIG.port, log_level="info")


if __name__ == "__main__":
    main()
