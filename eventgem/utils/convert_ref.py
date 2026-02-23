import struct
from pathlib import Path
import numpy as np

def read_header(f):
    magic, ver, H, W, off_top, off_left, K, D = struct.unpack("<4sIHHhhHH", f.read(4+4+2+2+2+2+2+2))
    assert magic == b"EKPR", f"bad magic: {magic}"
    return ver, H, W, off_top, off_left, K, D

def _read_exact(f, n: int) -> bytes | None:
    """Read exactly n bytes; return None if EOF/truncated."""
    b = f.read(n)
    if len(b) != n:
        return None
    return b

def main(raw_path: str, out_dir: str):
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "rb") as f:
        ver, H, W, off_top, off_left, K, D = read_header(f)

        rec_hdr_sz = struct.calcsize("<IqH")
        k_bytes = K * 2 * np.dtype(np.int16).itemsize
        s_bytes = K * np.dtype(np.float16).itemsize
        d_bytes = K * D * np.dtype(np.float16).itemsize

        i = 0
        while True:
            hdr = _read_exact(f, rec_hdr_sz)
            if hdr is None:
                break  # clean EOF or truncated header

            frame_idx, t_ref_raw, n_valid = struct.unpack("<IqH", hdr)

            k_buf = _read_exact(f, k_bytes)
            if k_buf is None:
                print(f"[WARN] Truncated file at record {i} (keypoints). Stopping.", flush=True)
                break

            s_buf = _read_exact(f, s_bytes)
            if s_buf is None:
                print(f"[WARN] Truncated file at record {i} (scores). Stopping.", flush=True)
                break

            d_buf = _read_exact(f, d_bytes)
            if d_buf is None:
                print(f"[WARN] Truncated file at record {i} (descriptors). Stopping.", flush=True)
                break

            # Now it's safe to parse
            k = np.frombuffer(k_buf, dtype=np.int16).reshape(K, 2)     # (y,x) cropped
            s = np.frombuffer(s_buf, dtype=np.float16)                 # scores
            d = np.frombuffer(d_buf, dtype=np.float16).reshape(K, D)   # desc

            n = int(n_valid)
            if n == 0:
                keypoints = np.empty((0, 2), np.float32)
                scores = np.empty((0,), np.float32)
                desc = np.empty((0, D), np.float32)
            else:
                k = k[:n].astype(np.float32)   # (y,x)
                k[:, 0] += off_top
                k[:, 1] += off_left
                keypoints = np.stack([k[:, 1], k[:, 0]], axis=-1).astype(np.float32)
                scores = s[:n].astype(np.float32)
                desc = d[:n].astype(np.float32)

            np.savez_compressed(
                out_dir / f"ref_kp_{frame_idx:05d}.npz",
                keypoints=keypoints,
                scores=scores,
                descriptors=desc,
                image_shape=np.array([H, W], dtype=np.int32),
            )
            i += 1