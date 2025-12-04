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

def recall(original, reranked, gt, k=[1, 5, 10, 20, 25]):
    """
    Computes Recall@K for original and re-ranked results.
    Args:
        original: List of lists containing original ranked indices for each query.
        reranked: List of lists containing re-ranked indices for each query.
        ks: List of K values for Recall@K computation.
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
    gt_tol = create_GTtol(gt, tolerance=200)

    table = PrettyTable()
    table.field_names = ["K", "Recall (Base)", "Recall (Geometric)"]
    
    for k in [1, 5, 10, 20, 25]:
        r_b = recallAtK((1-original), gt_tol, k)
        r_n = recallAtK((1-reranked), gt_tol, k)
        table.add_row([k, f"{r_b:.4f}", f"{r_n:.4f}"])

    print(table)