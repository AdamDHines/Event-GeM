import numpy as np

from prettytable import PrettyTable
from skimage.transform import resize
from eventgem.external.vprtutorial.evaluation.metrics import recallAtK

def create_GTtol(GT: np.ndarray, tolerance: int) -> np.ndarray:
    if tolerance <= 0: return GT
    GT = (GT > 0).astype(np.uint8)
    R, Q = GT.shape
    GTtol = GT.copy()
    ones = np.argwhere(GT > 0)
    for r, c in ones:
        r0, r1 = max(0, r - tolerance), min(R, r + tolerance + 1)
        c0, c1 = max(0, c - tolerance), min(Q, c + tolerance + 1)
        GTtol[r0:r1, c0:c1] = 1
    return GTtol

def recall(original, keypoint, depth, gt, k=[1, 5, 10]):
    """
    Computes Recall@K for original and re-ranked results.
    Args:
        original: List of lists containing original ranked indices for each query.
        keypoint: List of lists containing keypoint re-ranked indices for each query.
        depth: List of lists containing depth re-ranked indices for each query.
        gt: Ground truth array.
        k: List of K values for Recall@K computation.
    Returns:
        Two dictionaries with Recall@K values for original and re-ranked results.
    """
    # Check the balance of ratio of queries to references for original and gt
    ratio_original = original.shape[1] / original.shape[0]
    ratio_gt = gt.shape[1] / gt.shape[0]
    if not np.isclose(ratio_original, ratio_gt, atol=0.1):
        print(f"Warning: Ratio of queries to references in original results ({ratio_original:.2f}) "
              f"differs from that in ground truth ({ratio_gt:.2f}).")
    # Ensure shapes match
    if gt.shape != original.shape:
        gt = resize(gt, original.shape, order=0, preserve_range=True, anti_aliasing=False)
    # gt_tol = create_GTtol(gt, tolerance=200)

    ks = [1, 5, 10]

    # ---- Base always ----
    rec_base = {k: recallAtK((1 - original), gt, k) for k in ks}

    has_kp = keypoint is not None
    has_depth = depth is not None

    # ---- 1) Keypoint-only table ----
    if has_kp and not has_depth:
        table = PrettyTable()
        table.field_names = ["K", "Recall (Base)", "Recall (Keypoint)"]

        for k in ks:
            r_b = rec_base[k]
            r_kp = recallAtK((1 - keypoint), gt, k)
            table.add_row([k, f"{r_b:.4f}", f"{r_kp:.4f}"])

        print(table)

    # ---- 2) Depth-only table ----
    elif has_depth and not has_kp:
        table = PrettyTable()
        table.field_names = ["K", "Recall (Base)", "Recall (Depth)"]

        for k in ks:
            r_b = rec_base[k]
            r_d = recallAtK((1 - depth), gt, k)
            table.add_row([k, f"{r_b:.4f}", f"{r_d:.4f}"])

        print(table)

    # ---- 3) Both table ----
    elif has_kp and has_depth:
        table = PrettyTable()
        table.field_names = ["K", "Recall (Base)", "Recall (Keypoint)", "Recall (Depth)"]

        for k in ks:
            r_b = rec_base[k]
            r_kp = recallAtK((1 - keypoint), gt, k)
            r_d = recallAtK((1 - depth), gt, k)
            table.add_row([k, f"{r_b:.4f}", f"{r_kp:.4f}", f"{r_d:.4f}"])

        print(table)

    # ---- Nothing to compare ----
    else:
        table = PrettyTable()
        table.field_names = ["K", "Recall (Base)"]
        for k in ks:
            table.add_row([k, f"{rec_base[k]:.4f}"])
        print(table)