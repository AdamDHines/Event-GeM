"""
Single-file keypoint stores.

SuperEvent emits keypoints for every frame of a traverse -- >14k frames on brisbane_event -- and
the original path wrote one compressed `.npz` per frame, which the reranker then had to re-pack
into a dense bank before it could read it at speed. Accumulating the whole traverse in a Python
list first is not an option either: the descriptor block alone is ~2 GB per traverse
(N x 160 kpts x 256-D x 4 B), and two traverses are extracted back to back.

So the store is built the other way round: one padded block per component, memory-mapped from a
staging file and filled a batch at a time. Resident memory is one batch regardless of how long the
sequence is -- the kernel owns the rest and can evict it -- and the whole thing is serialised to a
single `<seq>_kps.pt` at the end, in the same spirit as `<seq>_features.pt`.

Layout (all CPU tensors, so the store loads on a machine without CUDA):

    descriptors  (N, K, D) float32   zero-padded to K = top_k
    keypoints    (N, K, 2) float32   (x, y) in *original* image coords
    scores       (N, K)    float32
    counts       (N,)      int32     valid rows per frame; the rest is padding
    image_shape  (2,)      int32     (H, W)

That is exactly the `(desc, kpts, counts)` bank shape `rerank_utils.bank_lookup` already consumes,
so the separate `build_reference_bank` re-pack is unnecessary when a store exists.
"""

import os

from pathlib import Path

import torch


def _mapped(path: Path, shape, dtype):
    """A zero-filled tensor of `shape` backed by `path` on disk rather than by RAM."""
    numel = 1
    for s in shape:
        numel *= int(s)
    return torch.from_file(str(path), shared=True, size=numel, dtype=dtype).view(*shape)


class KeypointStoreWriter:
    """
    Streams per-batch keypoints into one `.pt` store.

    Usage:
        writer = KeypointStoreWriter(out_path, n_frames, k_max, desc_dim, (H, W))
        try:
            for ...: writer.add_batch(kpts, desc, scores, counts)
            writer.save()
        finally:
            writer.cleanup()
    """

    def __init__(self, out_path, n_frames: int, k_max: int, desc_dim: int, image_shape):
        self.out_path = Path(out_path)
        self.n_frames = int(n_frames)
        self.k_max = int(k_max)
        self.desc_dim = int(desc_dim)
        self.image_shape = torch.tensor(list(image_shape), dtype=torch.int32)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = self.out_path.with_suffix("")
        self._staging = {name: Path(f"{stem}.{name}.tmp")
                         for name in ("descriptors", "keypoints", "scores")}
        for path in self._staging.values():
            if path.exists():
                path.unlink()

        self.descriptors = _mapped(self._staging["descriptors"],
                                   (self.n_frames, self.k_max, self.desc_dim), torch.float32)
        self.keypoints = _mapped(self._staging["keypoints"],
                                 (self.n_frames, self.k_max, 2), torch.float32)
        self.scores = _mapped(self._staging["scores"],
                              (self.n_frames, self.k_max), torch.float32)
        # Small enough to stay in RAM.
        self.counts = torch.zeros(self.n_frames, dtype=torch.int32)

        self.n_written = 0

    def add_batch(self, keypoints, descriptors, scores, counts):
        """
        Append one batch. All tensors are padded to k_max and must already be on the CPU:
        keypoints (B, K, 2), descriptors (B, K, D), scores (B, K), counts (B,).
        """
        b = int(counts.numel())
        i = self.n_written
        if i + b > self.n_frames:
            raise ValueError(f"keypoint store overflow: {i + b} frames written into a store "
                             f"sized for {self.n_frames}")

        self.descriptors[i:i + b] = descriptors
        self.keypoints[i:i + b] = keypoints
        self.scores[i:i + b] = scores
        self.counts[i:i + b] = counts.to(torch.int32)
        self.n_written += b

    def save(self):
        """Serialise the staged block to a single `.pt` and return its path."""
        if self.n_written != self.n_frames:
            # Trailing frames would otherwise be silently all-zero with count 0.
            print(f"[WARN] keypoint store {self.out_path.name}: wrote {self.n_written} of "
                  f"{self.n_frames} frames")

        torch.save(
            {
                "descriptors": self.descriptors,
                "keypoints": self.keypoints,
                "scores": self.scores,
                "counts": self.counts,
                "image_shape": self.image_shape,
            },
            self.out_path,
        )
        size_gb = os.path.getsize(self.out_path) / 1e9
        print(f"[INFO] keypoint store -> {self.out_path} "
              f"({self.n_written} frames, <={self.k_max} kpts x {self.desc_dim}-D, {size_gb:.2f} GB)")
        return self.out_path

    def cleanup(self):
        """Drop the mmaps and remove the staging files. Safe to call twice."""
        self.descriptors = self.keypoints = self.scores = None
        for path in self._staging.values():
            if path.exists():
                path.unlink()


def load_keypoint_store(path, mmap: bool = True):
    """
    Load a store written by `KeypointStoreWriter`.

    Memory-mapped by default so joblib workers share one copy through the page cache instead of
    each pickling their own ~2 GB.
    """
    return torch.load(str(path), map_location="cpu", mmap=mmap, weights_only=True)


def store_as_bank(store):
    """
    `(desc, kpts, counts)` in the tuple form `rerank_utils.bank_lookup` expects.

    `.numpy()` on a CPU tensor is a view, not a copy, so a memory-mapped store stays memory-mapped.
    """
    return (store["descriptors"].numpy(),
            store["keypoints"].numpy(),
            store["counts"].numpy())
