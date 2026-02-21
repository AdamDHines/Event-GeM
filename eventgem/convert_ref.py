import struct
from pathlib import Path
import numpy as np

def read_header(f):
    magic, ver, H, W, off_top, off_left, K, D = struct.unpack("<4sIHHhhHH", f.read(4+4+2+2+2+2+2+2))
    assert magic == b"EKPR", f"bad magic: {magic}"
    return ver, H, W, off_top, off_left, K, D

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
            hdr = f.read(rec_hdr_sz)
            if not hdr:
                break
            frame_idx, t_ref_raw, n_valid = struct.unpack("<IqH", hdr)

            k_buf = f.read(k_bytes)
            s_buf = f.read(s_bytes)
            d_buf = f.read(d_bytes)
            if len(k_buf) != k_bytes:
                break

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

                # convert to (x,y)
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

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])