import threading

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

from eventgem.utils.kp_store import load_keypoint_store, store_as_bank

# One memory-mapped store per path per process, opened lazily by whoever needs it first. Reranking
# runs on a thread pool, so this is shared rather than duplicated -- the lock only guards the
# first open. The store is opened by path rather than passed in because a mmapped tensor is not
# pickle-by-reference: handing the arrays to `delayed()` under a *process* backend would serialise
# the whole ~2 GB block per task.
_BANK_CACHE = {}
_BANK_LOCK = threading.Lock()


def open_keypoint_bank(store_path):
    """
    `(descriptors, keypoints, counts)` views onto a `<seq>_kps.pt` keypoint store.

    The store replaces the old one-npz-per-frame layout: it is already packed and padded to
    `k_max`, so there is nothing to re-pack -- `build_reference_bank`'s job is done at extraction
    time now. Returns None if the store is missing.
    """
    key = str(store_path)
    bank = _BANK_CACHE.get(key)
    if bank is None:
        with _BANK_LOCK:
            bank = _BANK_CACHE.get(key)
            if bank is None:
                if not Path(key).exists():
                    return None
                bank = _BANK_CACHE[key] = store_as_bank(load_keypoint_store(key))
    return bank


def bank_lookup(bank, idx):
    """One frame out of a keypoint store, in the {'kpts', 'desc'} form the matcher wants."""
    if bank is None:
        return None
    desc, kpts, counts = bank
    n = int(counts[idx])
    if n < 4:
        return None
    # np.array forces a real contiguous copy out of the mmap; cv2 will not take a mmap view.
    return {"desc": np.array(desc[idx, :n]), "kpts": np.array(kpts[idx, :n])}


def process_single_query(
    q_idx,
    base_dists,
    top_k,
    q_data,
    ref_store,
    ransac_thresh,
    inlier_weight,
    match_filter="ratio",
    match_ratio=0.8,
):
    """
    Rerank one query's shortlist. Worker function for the parallel loop.
    """
    # 1. Identify Candidates
    if top_k >= len(base_dists):
        top_indices = np.arange(len(base_dists))
    else:
        # Get indices of k smallest values (assuming distance matrix)
        part_indices = np.argpartition(base_dists, top_k)[:top_k]
        top_indices = part_indices[np.argsort(base_dists[part_indices])]

    new_dists_col = base_dists.copy()

    if q_data is None:
        return q_idx, new_dists_col

    # Opened once per process and shared by every worker thread.
    ref_bank = open_keypoint_bank(ref_store)
    if ref_bank is None:
        return q_idx, new_dists_col

    if match_filter == "mutual":
        _rerank_mutual(new_dists_col, base_dists, top_indices, q_data, ref_bank,
                       ransac_thresh, inlier_weight)
        return q_idx, new_dists_col

    # NORM_L2 for Float descriptors (SuperPoint/SuperEvent)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    for r_idx in top_indices:
        r_data = bank_lookup(ref_bank, r_idx)

        num_inliers = compute_inliers_2d(q_data, r_data, matcher, ransac_thresh,
                                         match_filter=match_filter, match_ratio=match_ratio)

        if num_inliers > 0:
            # Re-ranking logic:
            # NewDist = BaseDist - (Weight * Inliers)
            modifier = num_inliers * inlier_weight
            new_dists_col[r_idx] = base_dists[r_idx] - modifier

    return q_idx, new_dists_col


def _rerank_mutual(new_dists_col, base_dists, top_indices, q_data, ref_bank, ransac_thresh,
                   inlier_weight):
    """
    Mutual-NN rerank over one shortlist, written to run cheaply enough for a thread pool.

    Identical results to routing `match_filter="mutual"` through `mutual_nn_matches` and
    `compute_inliers_2d` -- verified column-for-column on 80 queries -- but ~1.8x faster
    single-threaded, from two observations:

    * The store's descriptors are exactly L2 normalised, so `||a-b||^2 = 2 - 2 a.b` and the
      nearest neighbour by Euclidean distance is the nearest by inner product. The norm terms,
      the broadcast add, the `maximum` and the `sqrt` that `mutual_nn_matches` computes -- about
      1.3 M elementwise operations per query at top_k=50 -- change no argmin and are skipped.
      argmin over distance becomes argmax over similarity.
    * The descriptor block goes to BLAS, not to cv2, so it does not need `bank_lookup`'s
      contiguous copy (~8 MB/query); each frame's slice of the mmap is already C-contiguous and
      can be multiplied in place. Only the matched keypoint rows are copied, for cv2.

    Profiled cost per query at top_k=50 before/after: 20.85 -> 11.51 ms sequential.
    """
    desc_bank, kpts_bank, counts = ref_bank
    desc_q, kpts_q = q_data["desc"], q_data["kpts"]
    nq = len(kpts_q)
    if nq < 4:
        return
    ar_nq = np.arange(nq)

    for r_idx in top_indices:
        n = int(counts[r_idx])
        if n < 4:
            continue

        sim = desc_q @ desc_bank[r_idx, :n].T          # (nq, n) inner products, no copy
        fwd = sim.argmax(axis=1)                       # query -> reference
        back = sim.argmax(axis=0)                      # reference -> query
        src = np.flatnonzero(back[fwd] == ar_nq)       # mutual pairs only
        if len(src) < 4:                               # a homography needs 4 points
            continue
        dst = fwd[src]

        src_pts = kpts_q[src].reshape(-1, 1, 2).astype(np.float32)
        dst_pts = kpts_bank[r_idx, dst].reshape(-1, 1, 2).astype(np.float32)
        try:
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_FAST, ransac_thresh,
                                         maxIters=100)
        except cv2.error:
            continue
        if mask is None:
            continue

        num_inliers = int(np.sum(mask))
        if num_inliers > 0:
            new_dists_col[r_idx] = base_dists[r_idx] - num_inliers * inlier_weight

def mutual_nn_matches(desc_q, desc_r):
    """
    Mutual nearest-neighbour correspondences: keep i <-> j only when j is i's nearest neighbour
    and i is j's.

    This replaces Lowe's ratio test for cross-condition event matching. With ~160 keypoints on a
    repetitive road scene the second nearest neighbour is routinely almost as close as the first,
    so the ratio test rejects correct matches wholesale -- measured at over 90% of ground-truth-true
    pairs on the morning and daytime traverses, which leaves fewer than the 4 points a homography
    needs and stops RANSAC running at all. Symmetry survives appearance change far better, and
    outlier rejection is left to RANSAC, which is what it is for.

    Returns (src_idx, dst_idx) index arrays.
    """
    # Squared L2 via expansion, then sqrt -- same metric as cv2.NORM_L2.
    d2 = ((desc_q * desc_q).sum(1)[:, None]
          + (desc_r * desc_r).sum(1)[None, :]
          - 2.0 * (desc_q @ desc_r.T))
    D = np.sqrt(np.maximum(d2, 0.0))

    nn_qr = D.argmin(axis=1)
    nn_rq = D.argmin(axis=0)
    src = np.flatnonzero(nn_rq[nn_qr] == np.arange(len(nn_qr)))
    return src, nn_qr[src]


def compute_inliers_2d(q_data, r_data, matcher, ransac_thresh, match_filter="ratio",
                       match_ratio=0.8):
    """
    Matches descriptors and runs 2D Homography RANSAC.
    Returns: number of inliers (int).
    """
    if q_data is None or r_data is None:
        return 0

    desc_q = q_data["desc"]
    desc_r = r_data["desc"]
    kpts_q = q_data["kpts"]
    kpts_r = r_data["kpts"]

    if len(kpts_q) < 4 or len(kpts_r) < 4:
        return 0

    # 1. Descriptor Matching
    if match_filter == "mutual":
        src_idx, dst_idx = mutual_nn_matches(desc_q, desc_r)
        if len(src_idx) < 4:
            return 0
        src_pts = kpts_q[src_idx].reshape(-1, 1, 2).astype(np.float32)
        dst_pts = kpts_r[dst_idx].reshape(-1, 1, 2).astype(np.float32)
    else:
        # SuperEvent descriptors are Float -> Use NORM_L2
        try:
            matches = matcher.knnMatch(desc_q, desc_r, k=2)
        except cv2.error:
            return 0

        good = []
        # Ratio Test (0.75 - 0.8 is standard)
        for m_n in matches:
            if len(m_n) == 2 and m_n[0].distance < match_ratio * m_n[1].distance:
                good.append(m_n[0])

        if len(good) < 4:
            return 0

        # 2. Extract matched points
        src_pts = np.float32([kpts_q[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts_r[m.trainIdx] for m in good]).reshape(-1, 1, 2)

    # 3. Geometric Verification (Homography + RANSAC)
    # RANSAC thresh: max reprojection error in pixels (e.g., 3.0 to 5.0)
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_FAST, ransac_thresh, maxIters=100)
    except cv2.error:
        return 0

    if mask is None:
        return 0
    
    # Sum the mask (1 = inlier, 0 = outlier)

    return int(np.sum(mask))