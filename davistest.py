#!/usr/bin/env python3
import time
from collections import deque
from datetime import timedelta

import numpy as np
import cv2
import dv_processing as dv


# ---------------------------
# DAVIS346 helpers (your generator, unchanged except imports)
# ---------------------------
def stream_event_windows_davis_live(dt_ms: float):
    cap = dv.io.camera.DAVIS()
    cap.setEventsRunning(True)
    cap.setFramesRunning(False)

    W, H = cap.getEventResolution()
    slicer = dv.EventStreamSlicer()
    q = deque()

    def cb(events):
        q.append(events)

    slicer.doEveryTimeInterval(timedelta(milliseconds=float(dt_ms)), cb)

    dt_us = int(round(float(dt_ms) * 1000.0))  # microseconds
    frame_idx = 0
    w_end_raw = None

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

            t_raw = np.asarray(ev.timestamps()).reshape(-1).astype(np.int64, copy=False)  # us
            p = np.asarray(ev.polarities()).reshape(-1).astype(np.uint8, copy=False)     # 0/1

            # Define window boundaries ourselves (stable even if ev has gaps)
            if w_end_raw is None:
                t0_raw = int(t_raw[0]) if t_raw.size else 0
                w_end_raw = t0_raw + dt_us
            else:
                w_end_raw += dt_us

            t_ref_raw = w_end_raw  # what MCTS expects

            yield (H, W, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms)
            frame_idx += 1


# ---------------------------
# Quick visualisation
# ---------------------------
def events_to_polarity_image(H, W, x, y, p):
    """
    Fast visual: accumulate +/-1 per pixel then map to gray.
    p expected 0/1 (0=neg, 1=pos).
    """
    img = np.zeros((H, W), dtype=np.int16)
    if x.size == 0:
        return np.full((H, W), 127, dtype=np.uint8)

    # pos += 1, neg -= 1
    # Do it with two indexed adds (fast + clear)
    pos = (p == 1)
    neg = ~pos

    if np.any(pos):
        np.add.at(img, (y[pos], x[pos]), 1)
    if np.any(neg):
        np.add.at(img, (y[neg], x[neg]), -1)

    # Clip for display; tweak these if you want more/less contrast
    img = np.clip(img, -5, 5)

    # Map [-5..5] -> [0..255], mid-gray = 127
    disp = ((img.astype(np.float32) + 5.0) / 10.0 * 255.0).astype(np.uint8)
    return disp


def visualise_davis346(dt_ms: float = 50.0, scale: float = 1.0):
    gen = stream_event_windows_davis_live(dt_ms)

    last_t = time.perf_counter()
    fps_ema = None

    win = "DAVIS346 events"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for (H, W, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms) in gen:
        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        inst_fps = 1.0 / max(dt, 1e-6)
        fps_ema = inst_fps if fps_ema is None else (0.9 * fps_ema + 0.1 * inst_fps)

        frame = events_to_polarity_image(H, W, x, y, p)

        if scale != 1.0:
            frame = cv2.resize(frame, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_NEAREST)

        # Overlay text
        overlay = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        text1 = f"dt={dt_ms:.1f}ms  fps~{fps_ema:.1f}  read={t_read_ms:.2f}ms"
        text2 = f"idx={frame_idx}  events={x.size}  t_ref(us)={t_ref_raw}"
        cv2.putText(overlay, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win, overlay)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualise_davis346(dt_ms=50.0, scale=1.5)
