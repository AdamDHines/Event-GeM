#!/usr/bin/env python3
"""
make_selector_npz.py

Builds an NPZ pack for selector_gate.py from:
  - sim0.npy: numpy array [N_ref, N_qry] (global similarity matrix)
  - gt.npy:   numpy array [*,*] resized to [N_ref, N_qry] via skimage.transform.resize

Optional:
  - sim_full.npy: numpy array [N_ref, N_qry] (full-pipeline similarity matrix) to compute top1_full
  - top1_full.npy: int array [N_qry] if you already computed full-pipeline top1 elsewhere

Output NPZ contains:
  topk_idx0, topk_val0, top1_full, gt_ptr, gt_ind, q_split
"""

from __future__ import annotations
import argparse
import os
import numpy as np
from skimage.transform import resize


def build_q_split(Nq: int, train_frac: float, val_frac: float, seed: int, mode: str) -> np.ndarray:
    assert 0 < train_frac < 1
    assert 0 <= val_frac < 1
    assert train_frac + val_frac < 1

    n_tr = int(round(Nq * train_frac))
    n_va = int(round(Nq * val_frac))
    n_te = Nq - n_tr - n_va

    split = np.empty(Nq, dtype=np.int8)

    if mode == "contiguous":
        split[:n_tr] = 0
        split[n_tr:n_tr + n_va] = 1
        split[n_tr + n_va:] = 2
    elif mode == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(Nq)
        split[perm[:n_tr]] = 0
        split[perm[n_tr:n_tr + n_va]] = 1
        split[perm[n_tr + n_va:]] = 2
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    return split


def resize_gt_to_match(gt: np.ndarray, shape_hw: tuple[int, int], thr: float) -> np.ndarray:
    """
    Resizes GT to shape [N_ref, N_qry] using skimage.transform.resize.
    Nearest-neighbour interpolation then threshold at thr.
    """
    N_ref, N_qry = shape_hw
    if gt.shape != (N_ref, N_qry):
        gt_r = resize(
            gt.astype(np.float32),
            (N_ref, N_qry),
            order=0,                 # nearest neighbour
            preserve_range=True,
            anti_aliasing=False,
        )
    else:
        gt_r = gt.astype(np.float32, copy=False)

    gt_bin = (gt_r > thr)
    return gt_bin


def gt_to_csr_by_query(gt_bin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert GT boolean matrix [N_ref, N_qry] into CSR-by-query:
      for each query q, store list of positive ref indices.
    Returns:
      gt_ptr int64 [Nq+1]
      gt_ind int32 [nnz]
    """
    N_ref, N_qry = gt_bin.shape

    # nnz per query (column)
    counts = gt_bin.sum(axis=0).astype(np.int64)  # [N_qry]

    gt_ptr = np.zeros(N_qry + 1, dtype=np.int64)
    gt_ptr[1:] = np.cumsum(counts)
    nnz = int(gt_ptr[-1])

    gt_ind = np.empty(nnz, dtype=np.int32)
    w = 0
    for q in range(N_qry):
        idx = np.flatnonzero(gt_bin[:, q]).astype(np.int32)  # sorted
        n = idx.size
        gt_ind[w:w + n] = idx
        w += n

    return gt_ptr, gt_ind


def compute_topk(sim: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    sim: float array [N_ref, N_qry]
    Returns:
      topk_idx0 int32 [N_qry, K]
      topk_val0 float32 [N_qry, K]
    Uses partial sort (argpartition) then sorts topK.
    """
    if sim.ndim != 2:
        raise ValueError(f"sim must be 2D [N_ref,N_qry], got {sim.shape}")
    N_ref, N_qry = sim.shape
    K = min(int(K), int(N_ref))

    # Argpartition over refs for each query (column-wise): work on transposed [N_qry, N_ref]
    S = sim.T  # [N_qry, N_ref]
    # indices of top-K (unordered)
    idx_part = np.argpartition(S, kth=N_ref - K, axis=1)[:, -K:]  # [N_qry, K]
    val_part = np.take_along_axis(S, idx_part, axis=1)            # [N_qry, K]

    # sort descending within top-K
    order = np.argsort(val_part, axis=1)[:, ::-1]
    topk_idx = np.take_along_axis(idx_part, order, axis=1).astype(np.int32, copy=False)
    topk_val = np.take_along_axis(val_part, order, axis=1).astype(np.float32, copy=False)

    return topk_idx, topk_val


def compute_top1_argmax(sim: np.ndarray) -> np.ndarray:
    """
    sim: float array [N_ref, N_qry]
    Returns top1 per query: int32 [N_qry]
    """
    return np.argmax(sim, axis=0).astype(np.int32, copy=False)


def quick_r1_from_gt_dense(gt_bin: np.ndarray, pred_top1: np.ndarray) -> float:
    """
    R@1 given dense gt_bin [N_ref,N_qry] and pred_top1 [N_qry].
    """
    N_ref, N_qry = gt_bin.shape
    q = np.arange(N_qry, dtype=np.int64)
    pred = pred_top1.astype(np.int64)
    pred = np.clip(pred, 0, N_ref - 1)
    return float(gt_bin[pred, q].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim0_npy", default="train_split/sunset2_sunset1_original.npy", help="Global similarity .npy [N_ref,N_qry]")
    ap.add_argument("--gt_npy", default="/media/adam/vprdatasets/eventgem/brisbane_event/ground_truth/sunset2_sunset1_GT.npy", help="GT .npy, resized to match sim0 shape")
    ap.add_argument("--out_npz", default="npz_pack", help="Output .npz pack")

    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--gt_threshold", type=float, default=0.5)

    # Optional full-pipeline top1 source:
    ap.add_argument("--sim_full_npy", default="train_split/sunset2_sunset1_reranked_depth.npy", help="Full similarity .npy [N_ref,N_qry]")
    ap.add_argument("--top1_full_npy", default=None, help="Precomputed top1_full .npy [N_qry]")

    # Splits:
    ap.add_argument("--split_mode", choices=["contiguous", "random"], default="contiguous")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)

    # ---- load sim0
    sim0 = np.load(args.sim0_npy, mmap_mode="r")
    sim0 = -np.asarray(sim0)  # IMPORTANT: distance -> similarity
    if sim0.ndim != 2:
        raise ValueError(f"sim0 must be 2D [N_ref,N_qry], got {sim0.shape}")
    N_ref, N_qry = sim0.shape
    print(f"[Sim0] {args.sim0_npy} shape=({N_ref},{N_qry}) dtype={sim0.dtype}")

    # ---- load + resize GT
    gt = np.load(args.gt_npy, mmap_mode=None)
    print(f"[GT]   {args.gt_npy} shape={gt.shape} dtype={gt.dtype} -> resizing to ({N_ref},{N_qry})")
    gt_bin = resize_gt_to_match(gt, (N_ref, N_qry), thr=args.gt_threshold)
    print(f"[GT]   bin positives={int(gt_bin.sum())} ({100.0*gt_bin.mean():.4f}% density)")

    # ---- compute topK from sim0
    topk_idx0, topk_val0 = compute_topk(np.asarray(sim0), K=args.K)
    top1_global = topk_idx0[:, 0].copy()
    print(f"[TopK] K={topk_idx0.shape[1]} computed")

    # ---- compute top1_full
    if args.top1_full_npy is not None:
        top1_full = np.load(args.top1_full_npy).astype(np.int32, copy=False)
        if top1_full.shape != (N_qry,):
            raise ValueError(f"top1_full_npy must be [N_qry], got {top1_full.shape}")
        print(f"[Full] top1_full loaded from {args.top1_full_npy}")
    elif args.sim_full_npy is not None:
        sim_full = np.load(args.sim_full_npy, mmap_mode="r")
        sim_full = -np.asarray(sim_full)  # IMPORTANT
        if sim_full.ndim != 2 or sim_full.shape != (N_ref, N_qry):
            raise ValueError(f"sim_full shape {sim_full.shape} != sim0 shape {(N_ref,N_qry)}")
        top1_full = compute_top1_argmax(np.asarray(sim_full))
        print(f"[Full] top1_full computed from {args.sim_full_npy}")
    else:
        top1_full = top1_global.copy()
        print(f"[Full] No sim_full_npy/top1_full_npy provided -> using global top1 as top1_full")

    # ---- CSR GT for selector_gate.py
    gt_ptr, gt_ind = gt_to_csr_by_query(gt_bin)
    print(f"[CSR]  nnz={gt_ind.size}  gt_ptr_len={gt_ptr.size}")

    # ---- build splits
    q_split = build_q_split(
        Nq=N_qry,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        mode=args.split_mode,
    )
    print(f"[Split] train={(q_split==0).sum()} val={(q_split==1).sum()} test={(q_split==2).sum()} mode={args.split_mode}")

    # ---- quick sanity R@1
    r1_global = quick_r1_from_gt_dense(gt_bin, top1_global)
    r1_full = quick_r1_from_gt_dense(gt_bin, top1_full)
    print(f"[Sanity] R@1_global={r1_global:.4f}  R@1_full={r1_full:.4f}")

    # ---- save
    np.savez_compressed(
        args.out_npz,
        topk_idx0=topk_idx0,
        topk_val0=topk_val0,
        top1_full=top1_full.astype(np.int32, copy=False),
        gt_ptr=gt_ptr,
        gt_ind=gt_ind,
        q_split=q_split,
    )
    print(f"[Saved] {args.out_npz}")


if __name__ == "__main__":
    main()