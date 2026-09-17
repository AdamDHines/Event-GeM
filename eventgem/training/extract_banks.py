"""Descriptor-bank building for the NYC contrastive head — eventcv MCTS only.

NYC frames come from megaevent's prepared samples (`prepare_nyc_event.py` output at
common.MEGAEVENT_NYC): 33.333 ms event windows cut from the RAW EVT3 archives, whose
headers carry the absolute recording epoch — so each sample's GPS fix is exact by
construction. MCTS only looks back `max_window_ms=30`, so a 33 ms sample renders the same
representation as a Brisbane 50 ms slice. Brisbane traverses render every 50 ms frame from
the production hdf5 + offset via `ecv.open(repr="mcts")`, the exact generation the
pipeline's dataset wraps; the p=5 columns reproduce the production descriptor
(--verify-against asserts it).

This module is a library first: train_head.py calls ensure_nyc_bank()/ensure_brisbane_bank()
to build whatever is missing, in-process — the banks directory is a cache, not a pipeline
artifact. The CLI remains for one-off builds and the faithfulness check:

  PYTHONPATH=. pixi run python -m eventgem.training.extract_banks --dataset brisbane \
      --sequences sunset2 --limit-frames 64 --verify-against <production bank .pt>
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

import eventcv as ecv
from eventgem.feature_extraction import EventGeM
from eventgem.training.common import (
    BRISBANE_OFFSETS, BRISBANE_ROWS, DESC_SLICES, MEGAEVENT_NYC, NYC_SENSOR_SIZE,
    crop_offsets, discover_nyc_recordings, haversine_m, pool_descriptor,
    pool_descriptor_v2,
)


def load_model(se_config_path, se_weights_path, device):
    """The frozen SuperEvent trunk + its config (yaml/weights loading reused from the
    pipeline; nothing else of the pipeline is involved here)."""
    eg = EventGeM(argparse.Namespace())
    return eg.build_superevent_model(Path(se_config_path), Path(se_weights_path), device)


def _atomic_savez(path, **arrays):
    tmp = str(path) + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _frame_prep(to_brisbane_scale):
    """Batch tensor -> trunk-ready tensor. `to_brisbane_scale` resizes to the DAVIS346
    geometry the trunk was trained at (height 260, aspect kept, centre-cropped to 346)."""
    import torch.nn.functional as TF

    def prep(x):
        if to_brisbane_scale:
            h, w = x.shape[-2:]
            tw = max(346, round(w * 260 / h))   # height -> 260, aspect preserved
            x = TF.interpolate(x, size=(260, tw), mode="bilinear", antialias=True)
            off = (tw - 346) // 2               # centre-crop width to 346
            x = x[:, :, :, off:off + 346]
        return x
    return prep


def _pool_batches(frame_iter, n_frames, model, se_config, device, batch_size, tag,
                  to_brisbane_scale=False, pool_fn=None):
    """MCTS frames from `frame_iter` through the frozen trunk -> [n_frames, D].

    `pool_fn` defaults to the v1 1024-D `pool_descriptor`; pass `pool_descriptor_v2` to
    build a v2 superset bank instead (2026-08-27 exponent factorial). The trunk pass,
    the frame prep and the crop are identical either way -- only the pooling differs."""
    pool_fn = pool_fn or pool_descriptor
    prep = _frame_prep(to_brisbane_scale)
    descs, buf, t0, done = [], [], time.time(), 0
    crop = None

    def flush():
        nonlocal crop, done
        if not buf:
            return
        x = prep(torch.from_numpy(np.stack(buf)))
        if crop is None:
            _, _, H, W = x.shape
            crop = crop_offsets(H, W, se_config)
        ot, he, ol, we = crop
        with torch.inference_mode():
            feats = model.fpn(model.backbone(x[:, :, ot:he, ol:we].to(device))).float()
            descs.append(pool_fn(feats).cpu())
        done += len(buf)
        buf.clear()
        if (done // batch_size) % 25 == 0:
            print(f"[{tag}] {done}/{n_frames} frames "
                  f"({done / max(time.time() - t0, 1e-6):.1f} f/s)", flush=True)

    for frame in frame_iter:
        buf.append(frame)
        if len(buf) >= batch_size:
            flush()
    flush()
    desc = torch.cat(descs).numpy().astype(np.float32)
    assert np.isfinite(desc).all(), f"[{tag}] non-finite descriptors"
    return desc


def _nyc_frame(npz_path):
    """One megaevent sample -> a 10-channel MCTS frame (float32 [10, 720, 1280])."""
    with np.load(npz_path) as z:
        n = z["x"].shape[0]
        arr = np.empty((n, 4), dtype=np.int64)
        arr[:, 0], arr[:, 1] = z["x"], z["y"]
        arr[:, 2], arr[:, 3] = z["t"], z["p"]
    if n == 0:
        return np.zeros((10, NYC_SENSOR_SIZE[1], NYC_SENSOR_SIZE[0]), dtype=np.float32)
    s = ecv.from_numpy(arr, time_unit="us", sensor_size=NYC_SENSOR_SIZE).hot_pixel_filter()
    if len(s) == 0:
        return np.zeros((10, NYC_SENSOR_SIZE[1], NYC_SENSOR_SIZE[0]), dtype=np.float32)
    return s.mcts().numpy()


def ensure_nyc_bank(name, rows, out_dir, model, se_config, device,
                    batch_size=8, min_speed=0.0, limit_frames=0, overwrite=False,
                    to_brisbane_scale=False, pool_fn=None):
    """Build (or reuse) the descriptor bank for one NYC traverse from megaevent samples.

    Banks carry every sample (min_speed defaults to 0 since the campaign rework):
    stationary-duplicate handling moved to train time (displacement dedup), so a bank
    never has to be rebuilt to change the filtering policy. `heading` (degrees, from the
    manifest) is stored per sample for direction-gated positives.
    """
    out_path = Path(out_dir) / f"nyc_{name}.npz"
    if out_path.exists() and not overwrite:
        return out_path
    t = np.array([r["t_wall_s"] for r in rows])
    lat = np.array([r["lat"] for r in rows])
    lon = np.array([r["lon"] for r in rows])
    heading = np.array([r["heading"] for r in rows])

    if min_speed > 0:
        dt = np.diff(t)
        step = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        speed = np.full(len(rows), np.nan)
        ok = (dt > 0.2) & (dt < 10.0)
        speed[1:][ok] = step[ok] / dt[ok]
        keep = np.flatnonzero(speed >= min_speed)
    else:
        keep = np.arange(len(rows))
    if limit_frames:
        keep = keep[:limit_frames]
    print(f"[{name}] {len(keep)}/{len(rows)} samples kept (min_speed={min_speed})")
    if len(keep) == 0:
        raise ValueError(f"[{name}] no samples to extract")

    frames = (_nyc_frame(rows[i]["path"]) for i in keep)
    desc = _pool_batches(frames, len(keep), model, se_config, device, batch_size, name,
                         to_brisbane_scale=to_brisbane_scale, pool_fn=pool_fn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(out_path, desc=desc, lat=lat[keep], lon=lon[keep],
                  heading=heading[keep], t_epoch=t[keep], recording=name)
    print(f"[{name}] wrote {out_path} desc {desc.shape}")
    return out_path


def ensure_brisbane_bank(seq, bris_root, out_dir, model, se_config, device,
                         batch_size=16, limit_frames=0, overwrite=False,
                         verify_against=None, pool_fn=None):
    """Build (or reuse) the descriptor bank for one Brisbane traverse — every 50 ms frame,
    framed by the production offset, straight from the eventcv reader."""
    out_path = Path(out_dir) / f"brisbane_{seq}.npz"
    if out_path.exists() and not overwrite:
        return out_path
    hdf5 = f"{bris_root}/{seq}/{seq}.hdf5"
    reader = ecv.open(hdf5, dt_ms=50, offset=BRISBANE_OFFSETS[seq],
                      hot_pixel_filter=True, repr="mcts")
    n = reader.n_slices
    if not limit_frames and seq in BRISBANE_ROWS:
        assert n == BRISBANE_ROWS[seq], f"{seq}: {n} slices vs expected {BRISBANE_ROWS[seq]}"
    indices = range(min(n, limit_frames) if limit_frames else n)
    frames = (reader[int(i)] for i in indices)
    desc = _pool_batches(frames, len(indices), model, se_config, device, batch_size, seq,
                         pool_fn=pool_fn)

    if verify_against:
        assert pool_fn is None, "verify_against reads the v1 p=5 slice; only valid for v1 banks"
        bank = torch.load(verify_against, map_location="cpu").float().numpy()
        m = min(len(desc), len(bank))
        a, b = desc[:m, DESC_SLICES["p5"]], bank[:m]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        print(f"[{seq}] verify vs {verify_against}: min cosine {cos.min():.8f} over {m} rows")
        assert cos.min() > 1 - 1e-5, "p=5 columns do not reproduce the production bank"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(out_path, desc=desc, recording=seq)
    print(f"[{seq}] wrote {out_path} desc {desc.shape}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["nyc", "brisbane"], required=True)
    ap.add_argument("--out",
                    default=os.path.join(os.environ.get("EVENTGEM_OUT", "."), "banks"))
    ap.add_argument("--nyc-root", default=os.environ.get("EVENTGEM_NYC_ROOT", MEGAEVENT_NYC))
    ap.add_argument("--bris-root", default=os.environ.get(
        "EVENTGEM_BRIS_ROOT", "/media/adam/vprdatasets/eventgem/brisbane_event"))
    ap.add_argument("--recording", default=None, help="NYC traverse name (default: all)")
    ap.add_argument("--sequences", nargs="+", default=["sunset2", "sunset1"])
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="extraction-time speed filter; 0 keeps everything (default — "
                         "dedup happens at train time)")
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--to-brisbane-scale", action="store_true",
                    help="resize NYC frames to DAVIS346 geometry before the trunk")
    ap.add_argument("--verify-against", default=None)
    ap.add_argument("--se-config",
                    default="eventgem/external/superevent/config/super_event.yaml")
    ap.add_argument("--se-weights",
                    default="eventgem/external/superevent/saved_models/super_event_weights.pth")
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = 8 if args.dataset == "nyc" else 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, se_config = load_model(args.se_config, args.se_weights, device)
    if args.dataset == "nyc":
        recs = discover_nyc_recordings(args.nyc_root)
        if args.recording:
            recs = [r for r in recs if r[0] == args.recording]
            if not recs:
                raise SystemExit(f"recording {args.recording!r} not in manifest")
        for name, rows in recs:
            ensure_nyc_bank(name, rows, args.out, model, se_config, device,
                            args.batch_size, args.min_speed, args.limit_frames,
                            args.overwrite, to_brisbane_scale=args.to_brisbane_scale)
    else:
        for seq in args.sequences:
            ensure_brisbane_bank(seq, args.bris_root, args.out, model, se_config, device,
                                 args.batch_size, args.limit_frames, args.overwrite,
                                 args.verify_against)


if __name__ == "__main__":
    main()
