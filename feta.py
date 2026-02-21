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
    if len(z.files) == 1:
        arr = z[z.files[0]]
    else:
        # common guesses; if none match, take first
        for k in ("desc", "q_desc_vit", "feat", "feats", "descriptor", "descriptors", "emb", "embedding"):
            if k in z.files:
                arr = z[k]
                break
        else:
            arr = z[z.files[0]]

    arr = np.asarray(arr)
    # Make it 1D [D]
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim != 1:
        arr = arr.reshape(-1)

    return arr.astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="ref_feats_*.npz")
    ap.add_argument("--out", type=str, required=True, help="Output .pt (tensor only)")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    paths = sorted(npz_dir.glob(args.pattern), key=frame_key)
    if not paths:
        raise SystemExit(f"No files matched {args.pattern} in {npz_dir}")

    descs = [load_desc(p) for p in paths]

    D0 = descs[0].shape[0]
    for p, d in zip(paths, descs):
        if d.shape[0] != D0:
            raise ValueError(f"Dim mismatch: {p.name} has D={d.shape[0]} expected D={D0}")

    mat = torch.from_numpy(np.stack(descs, axis=0))  # [N, D], float32

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mat, out_path)
    print(f"Saved {out_path} with shape {tuple(mat.shape)} dtype={mat.dtype}")


if __name__ == "__main__":
    main()
