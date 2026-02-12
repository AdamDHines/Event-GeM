from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
import argparse
import numpy as np
import prettytable
from skimage.transform import resize
import cv2

def recallAtK(S, GT, K=1):
    """
    Calculates the recall@K for a given similarity matrix S and ground truth matrix GT.
    Note that this method does not support GTsoft - instead, please directly provide
    the dilated ground truth matrix as GT.

    The matrices S and GT are two-dimensional and should all have the same shape.
    The matrix GT should be binary, where the entries are only zeros or ones.
    The matrix S should have continuous values between -Inf and Inf. Higher values
    indicate higher similarity.
    The integer K>=1 defines the number of matching candidates that are selected and
    that must contain an actually matching image pair.
    """

    assert (S.shape == GT.shape),"S and GT must have the same shape"
    assert (S.ndim == 2),"S and GT must be two-dimensional"
    assert (K >= 1),"K must be >=1"

    # ensure logical datatype in GT
    GT = GT.astype('bool')

    # discard all query images without an actually matching database image
    j = GT.sum(0) > 0 # columns with matches
    S = S[:,j] # select columns with a match
    GT = GT[:,j] # select columns with a match

    # select K highest similarities
    i = S.argsort(0)[-K:,:]
    j = np.tile(np.arange(i.shape[1]), [K, 1])
    GT = GT[i, j]

    # recall@K
    RatK = np.sum(GT.sum(0) > 0) / GT.shape[1]

    return RatK

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

def rerank(distance_matrix_path, reference_keypoints, query_keypoints, ground_truth, output, top_k=100, 
           ransac_thresh=5.0, inlier_weight=0.5, k=[1,5,10], kp_pattern="mcts_{:05d}.feat.npz"):
    distance_matrix = np.load(distance_matrix_path)
    # distance_matrix = 1-distance_matrix
    R, Q = distance_matrix.shape
    # load the ground truth file
    gt = np.load(ground_truth)
    # Pre-load queries exactly like before
    queries_data = []
    for i in tqdm(range(Q)):
        queries_data.append(
            load_event_features(Path(query_keypoints), i, kp_pattern)
        )

    dist_matrix = distance_matrix
    results = Parallel(n_jobs=-1)(
        delayed(process_single_query)(
            q_idx=i,
            base_dists=dist_matrix[:, i],
            top_k=top_k,
            q_data=queries_data[i],
            ref_kp_dir=Path(reference_keypoints),
            kp_pattern=kp_pattern,
            ransac_thresh=ransac_thresh,
            inlier_weight=inlier_weight,
        )
        for i in tqdm(range(Q), desc="Re-ranking")
    )

    new_dist_matrix = dist_matrix.copy()
    for q_idx, new_col in results:
        new_dist_matrix[:, q_idx] = new_col
    new_dist_matrix = 1-new_dist_matrix
    distance_matrix = 1-distance_matrix

    print("Distance shape:", new_dist_matrix.shape)
    print("GT shape:", gt.shape)

    # reshape the ground truth matrix if needed
    if new_dist_matrix.shape != gt.shape:
        gt = resize(gt, distance_matrix.shape, order=0, preserve_range=True, anti_aliasing=False)

    # Evaluate recall before and after re-ranking
    table = prettytable.PrettyTable()
    table.field_names = ["K", "Recall Before Rerank", "Recall After Rerank"]
    for k_val in k:
        pre_ranks = recallAtK(distance_matrix, gt, k_val)
        post_ranks = recallAtK(new_dist_matrix, gt, k_val)
        table.add_row([k_val, f"{pre_ranks:.4f}", f"{post_ranks:.4f}"])
    print(table)

    # np.save(output, new_dist_matrix)

def main():
    parser = argparse.ArgumentParser(
        description="Re-rank event-based VPR results using geometric verification."
    )
    parser.add_argument(
        "--dist-matrix",
        type=str,
        required=True,
        help="Path to the initial distance matrix (numpy .npy file).",
    )
    parser.add_argument(
        "--reference-keypoints",
        type=str,
        required=True,
        help="Path to the reference keypoints directory.",
    )
    parser.add_argument(
        "--query-keypoints",
        type=str,
        required=True,
        help="Path to the query keypoints directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of top candidates to re-rank.",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=5.0,
        help="RANSAC inlier threshold.",
    )
    parser.add_argument(
        "--inlier-weight",
        type=float,
        default=0.5,
        help="Weight for inlier count in final score.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the re-ranked distance matrix (numpy .npy file).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to the ground truth file (numpy .npy file).",
    )

    args = parser.parse_args()

    reranker = rerank(
        args.dist_matrix,
        args.reference_keypoints,
        args.query_keypoints,
        args.ground_truth,
        args.output,
        top_k=args.top_k,
        ransac_thresh=args.ransac_thresh,
        inlier_weight=args.inlier_weight,
    )

if __name__ == "__main__":
    main()