import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

def load_event_features(root_dir: Path, idx: int, pattern: str):
    """
    Loads keypoints and descriptors from EventGlue output (.feat.npz).
    Expects keys: 'keypoints' (N,2), 'descriptors' (N,D).
    """
    # Pattern example: "mcts_{:04d}.feat.npz" or just "*.feat.npz" logic
    fname = pattern.format(idx)
    path = root_dir / fname
    
    if not path.exists():
        return None

    try:
        data = np.load(str(path))
        
        # Check for keys (EventGlue/SuperPoint format)
        if "keypoints" in data:
            kpts = data["keypoints"].astype(np.float32)
        elif "x" in data and "y" in data:
            kpts = np.stack([data["x"], data["y"]], axis=1).astype(np.float32)
        else:
            return None

        if "descriptors" in data:
            desc = data["descriptors"].astype(np.float32)
        elif "desc" in data:
            desc = data["desc"].astype(np.float32)
        else:
            return None

        # Handle empty features
        if kpts.shape[0] == 0 or desc.shape[0] == 0:
            return None
            
        # Ensure dimensions
        if desc.ndim == 1:
            desc = desc.reshape(1, -1)

        return {"kpts": kpts, "desc": desc}

    except Exception:
        return None

def build_reference_bank(ref_kp_dir, n_refs, kp_pattern, rebuild=False):
    """
    Pack a reference keypoint store into dense .npy files and return them memory-mapped.

    Reranking is I/O bound, not compute bound: loading one compressed `.npz` per candidate costs
    ~0.884 ms against ~0.253 ms for the actual descriptor matching, i.e. 80% of the per-pair budget
    is spent decompressing files. Packing the store into two flat arrays turns that into a memcpy
    out of the OS page cache.

    The pack is written next to the store and reused on later runs, so it is paid once. It is
    memory-mapped rather than loaded, so joblib workers share one copy of the ~2 GB via the page
    cache instead of each pickling their own.

    Returns (desc, kpts, counts) or None if the store cannot be packed.
    """
    ref_kp_dir = Path(ref_kp_dir)
    stem = ref_kp_dir.parent / f"{ref_kp_dir.name}_bank"
    desc_path = Path(f"{stem}_desc.npy")
    kpts_path = Path(f"{stem}_kpts.npy")
    counts_path = Path(f"{stem}_counts.npy")

    if rebuild or not (desc_path.exists() and kpts_path.exists() and counts_path.exists()):
        probe = None
        for i in range(n_refs):
            probe = load_event_features(ref_kp_dir, i, kp_pattern)
            if probe is not None:
                break
        if probe is None:
            print(f"[WARN] no keypoint files under {ref_kp_dir}; not packing")
            return None

        k_max, dim = probe["desc"].shape
        print(f"[INFO] packing {n_refs} reference frames -> {desc_path.name} "
              f"({n_refs * k_max * dim * 4 / 1e9:.2f} GB)", flush=True)

        desc = np.lib.format.open_memmap(desc_path, mode="w+", dtype=np.float32,
                                         shape=(n_refs, k_max, dim))
        kpts = np.lib.format.open_memmap(kpts_path, mode="w+", dtype=np.float32,
                                         shape=(n_refs, k_max, 2))
        counts = np.zeros(n_refs, dtype=np.int32)

        for i in tqdm(range(n_refs), desc="Packing reference keypoints", unit="frame"):
            d = load_event_features(ref_kp_dir, i, kp_pattern)
            if d is None:
                continue
            n = min(len(d["kpts"]), k_max)
            desc[i, :n] = d["desc"][:n]
            kpts[i, :n] = d["kpts"][:n]
            counts[i] = n

        desc.flush()
        kpts.flush()
        np.save(counts_path, counts)
        del desc, kpts

    return (np.load(desc_path, mmap_mode="r"),
            np.load(kpts_path, mmap_mode="r"),
            np.load(counts_path))


def bank_lookup(ref_bank, idx):
    """One reference frame out of the packed bank, in load_event_features' dict format."""
    desc, kpts, counts = ref_bank
    n = int(counts[idx])
    if n < 4:
        return None
    # np.array forces a real contiguous copy out of the memmap; cv2 will not take a mmap view.
    return {"desc": np.array(desc[idx, :n]), "kpts": np.array(kpts[idx, :n])}


def process_single_query(
    q_idx,
    base_dists,
    top_k,
    q_data,
    ref_kp_dir,
    kp_pattern,
    ransac_thresh,
    inlier_weight,
    match_filter="ratio",
    match_ratio=0.8,
    ref_bank=None
):
    """
    Worker function for parallel processing.
    """
    # 1. Identify Candidates
    if top_k >= len(base_dists):
        top_indices = np.arange(len(base_dists))
    else:
        # Get indices of k smallest values (assuming distance matrix)
        part_indices = np.argpartition(base_dists, top_k)[:top_k]
        top_indices = part_indices[np.argsort(base_dists[part_indices])]

    # 2. Init Matcher (Local to thread)
    # NORM_L2 for Float descriptors (SuperPoint/SuperEvent)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # 3. Process
    new_dists_col = base_dists.copy()

    if q_data is None:
        return q_idx, new_dists_col

    for r_idx in top_indices:
        # Packed bank if available (memmap slice), otherwise a per-candidate npz read
        if ref_bank is not None:
            r_data = bank_lookup(ref_bank, r_idx)
        else:
            r_data = load_event_features(ref_kp_dir, r_idx, kp_pattern)


        num_inliers = compute_inliers_2d(q_data, r_data, matcher, ransac_thresh,
                                         match_filter=match_filter, match_ratio=match_ratio)

        if num_inliers > 0:
            # Re-ranking logic:
            # NewDist = BaseDist - (Weight * Inliers)
            modifier = num_inliers * inlier_weight
            new_dists_col[r_idx] = base_dists[r_idx] - modifier

    return q_idx, new_dists_col

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