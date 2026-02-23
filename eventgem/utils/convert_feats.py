#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import torch


def frame_key(path: Path) -> int:
    m = re.search(r"ref_feats_(\d+)\.npz$", path.name)
    return int(m.group(1)) if m else 10**18  # shove weird names to the end


def load_desc(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=False)

    # If you saved a named array (recommended), set this key to that name.
    # Otherwise we fall back to the first array in the file (often "arr_0").
    arr = z['arr_0']

    arr = np.asarray(arr)
    # Make it 1D [D]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 1:
        arr = arr.reshape(-1)

    return arr.astype(np.float32, copy=False)


def main(npy_dir, out, pattern="ref_feats_*.npz"):
    npy_dir = Path(npy_dir)
    paths = sorted(npy_dir.glob(pattern), key=frame_key)
    if not paths:
        raise SystemExit(f"No files matched {pattern} in {npy_dir}")

    descs = [load_desc(p) for p in paths]

    D0 = descs[0].shape[0]
    for p, d in zip(paths, descs):
        if d.shape[0] != D0:
            raise ValueError(f"Dim mismatch: {p.name} has D={d.shape[0]} expected D={D0}")

    mat = torch.from_numpy(np.stack(descs, axis=0))  # [N, D], float32

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mat, out_path)
    print(f"Saved {out_path} with shape {tuple(mat.shape)} dtype={mat.dtype}")

    # delete all .npz files after conversion from the operating system
    for p in paths:
        p.unlink()
    print(f"Deleted {len(paths)} .npy files in {npy_dir}")