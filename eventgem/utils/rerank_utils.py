import numpy as np
import cv2
from pathlib import Path
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

def process_single_query(
    q_idx, 
    base_dists, 
    top_k, 
    q_data, 
    ref_kp_dir, 
    kp_pattern, 
    ransac_thresh,
    inlier_weight
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
        # Load ref data on demand
        r_data = load_event_features(ref_kp_dir, r_idx, kp_pattern)
        
        num_inliers = compute_inliers_2d(q_data, r_data, matcher, ransac_thresh)

        if num_inliers > 0:
            # Re-ranking logic:
            # NewDist = BaseDist - (Weight * Inliers)
            modifier = num_inliers * inlier_weight
            new_dists_col[r_idx] = base_dists[r_idx] - modifier

    return q_idx, new_dists_col

def compute_inliers_2d(q_data, r_data, matcher, ransac_thresh):
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
    # SuperEvent descriptors are Float -> Use NORM_L2
    try:
        matches = matcher.knnMatch(desc_q, desc_r, k=2)
    except cv2.error:
        return 0

    good = []
    # Ratio Test (0.75 - 0.8 is standard)
    for m_n in matches:
        if len(m_n) == 2 and m_n[0].distance < 0.8 * m_n[1].distance:
            good.append(m_n[0])

    if len(good) < 4:
        return 0

    # 2. Extract matched points
    src_pts = np.float32([kpts_q[m.queryIdx] for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts_r[m.trainIdx] for m in good]).reshape(-1, 1, 2)

    # 3. Geometric Verification (Homography + RANSAC)
    # RANSAC thresh: max reprojection error in pixels (e.g., 3.0 to 5.0)
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    except cv2.error:
        return 0

    if mask is None:
        return 0
    
    # Sum the mask (1 = inlier, 0 = outlier)
    return int(np.sum(mask))