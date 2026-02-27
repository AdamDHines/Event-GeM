#!/usr/bin/env python3
"""
selector_gate.py
Uncertainty-aware selective reranking gate (deep ensemble MLP + temp scaling).

Data format (NPZ) expected per experiment:
  topk_idx0:  int32 [Nq, K]    global top-K reference indices (sorted by similarity desc)
  topk_val0:  float32 [Nq, K]  global top-K similarities (sorted desc)
  top1_full:  int32 [Nq]       full-pipeline predicted top-1 ref index per query

  gt_ptr:     int64 [Nq+1]     CSR pointer for GT positives
  gt_ind:     int32 [nnz]      CSR indices for GT positives

Optional:
  q_split:    int8  [Nq]       0=train,1=val,2=test (if you want single-file split)
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# GT utilities (CSR positives)
# -----------------------------

def gt_is_positive(gt_ptr: np.ndarray, gt_ind: np.ndarray, q: int, ref_idx: int) -> bool:
    """Check if ref_idx is in the GT positives for query q (CSR row membership)."""
    a = int(gt_ptr[q]); b = int(gt_ptr[q + 1])
    # gt_ind row is sorted? If yes, we can binary search; if not, fallback to linear.
    row = gt_ind[a:b]
    # Try binary search if sorted:
    # If you aren't sure it's sorted, just use linear membership for now.
    # return ref_idx in row
    i = np.searchsorted(row, ref_idx)
    return (i < row.size) and (int(row[i]) == int(ref_idx))


def top1_correct_from_topk(gt_ptr, gt_ind, topk_idx0: np.ndarray) -> np.ndarray:
    """Compute correctness of global top-1 from topk indices."""
    Nq = topk_idx0.shape[0]
    pred = topk_idx0[:, 0]
    out = np.zeros(Nq, dtype=np.bool_)
    for q in range(Nq):
        out[q] = gt_is_positive(gt_ptr, gt_ind, q, int(pred[q]))
    return out


def top1_correct_from_pred(gt_ptr, gt_ind, pred_top1: np.ndarray) -> np.ndarray:
    """Compute correctness for provided top-1 predictions."""
    Nq = pred_top1.shape[0]
    out = np.zeros(Nq, dtype=np.bool_)
    for q in range(Nq):
        out[q] = gt_is_positive(gt_ptr, gt_ind, q, int(pred_top1[q]))
    return out


# -----------------------------
# Feature extraction
# -----------------------------

def softmax_entropy(v: np.ndarray, temp: float = 1.0) -> float:
    """Entropy of softmax(v/temp)."""
    x = v / max(temp, 1e-6)
    x = x - x.max()
    p = np.exp(x)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def extract_features_from_topk(topk_val0: np.ndarray) -> np.ndarray:
    """
    Features from sorted top-K similarity values (desc).
    Returns float32 [Nq, D].
    Minimal but decent: raw top-K + margins + moments + entropy/pmax.
    """
    v = topk_val0
    Nq, K = v.shape

    v1 = v[:, 0]
    v2 = v[:, 1] if K > 1 else v[:, 0]
    v5 = v[:, 4] if K > 4 else v[:, -1]
    v10 = v[:, 9] if K > 9 else v[:, -1]

    margin12 = v1 - v2
    margin15 = v1 - v5
    margin110 = v1 - v10

    mu = v.mean(axis=1)
    sig = v.std(axis=1) + 1e-6
    z1 = (v1 - mu) / sig

    # entropy and pmax over top-K (numpy loop; fine for iteration; optimize later if needed)
    ent = np.zeros(Nq, dtype=np.float32)
    pmax = np.zeros(Nq, dtype=np.float32)
    for i in range(Nq):
        x = v[i] - v[i].max()
        p = np.exp(x)
        p = p / (p.sum() + 1e-12)
        ent[i] = float(-(p * np.log(p + 1e-12)).sum())
        pmax[i] = float(p.max())

    # Concatenate: [raw topK] + stats
    feats = np.concatenate([
        v.astype(np.float32),
        v1[:, None].astype(np.float32),
        v2[:, None].astype(np.float32),
        v5[:, None].astype(np.float32),
        v10[:, None].astype(np.float32),
        margin12[:, None].astype(np.float32),
        margin15[:, None].astype(np.float32),
        margin110[:, None].astype(np.float32),
        mu[:, None].astype(np.float32),
        sig[:, None].astype(np.float32),
        z1[:, None].astype(np.float32),
        ent[:, None],
        pmax[:, None],
    ], axis=1)

    return feats.astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------

@dataclass
class SelectorPack:
    topk_idx0: np.ndarray
    topk_val0: np.ndarray
    top1_full: np.ndarray
    gt_ptr: np.ndarray
    gt_ind: np.ndarray
    split: Optional[np.ndarray] = None  # 0/1/2

    @property
    def Nq(self) -> int:
        return int(self.topk_idx0.shape[0])

    @property
    def K(self) -> int:
        return int(self.topk_idx0.shape[1])


def load_pack(npz_path: str) -> SelectorPack:
    z = np.load(npz_path, allow_pickle=False)
    pack = SelectorPack(
        topk_idx0=z["topk_idx0"].astype(np.int32),
        topk_val0=z["topk_val0"].astype(np.float32),
        top1_full=z["top1_full"].astype(np.int32),
        gt_ptr=z["gt_ptr"].astype(np.int64),
        gt_ind=z["gt_ind"].astype(np.int32),
        split=z["q_split"].astype(np.int8) if "q_split" in z else None,
    )
    return pack


class TopKSelectorDataset(Dataset):
    def __init__(
        self,
        feats: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
    ):
        self.feats = feats
        self.y = y.astype(np.float32)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return torch.from_numpy(self.feats[idx]), torch.tensor(self.y[idx])


def make_benefit_labels(pack: SelectorPack) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      correct0: bool [Nq]
      correct_full: bool [Nq]
      y_benefit: bool [Nq]  (global wrong AND full correct)
    """
    correct0 = top1_correct_from_topk(pack.gt_ptr, pack.gt_ind, pack.topk_idx0)
    correct_full = top1_correct_from_pred(pack.gt_ptr, pack.gt_ind, pack.top1_full)
    y = correct0  # predict "good enough"
    return correct0, correct_full, y


# -----------------------------
# Model
# -----------------------------

class MLPSelector(nn.Module):
    def __init__(self, d_in: int, h1: int = 128, h2: int = 64, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d_in), # Added for scale-invariant features
            nn.Linear(d_in, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle single-sample inference (BatchNorm requires >1 batch or eval mode)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x).squeeze(-1)


class TemperatureScaler(nn.Module):
    """Single scalar temperature for calibration."""
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(math.log(init_temp), dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = torch.exp(self.log_t).clamp(1e-3, 1e3)
        return logits / t


# -----------------------------
# Training / eval helpers
# -----------------------------

@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in loader:
        xb = xb.to(device)
        logit = model(xb).detach().cpu().numpy()
        out.append(logit)
    return np.concatenate(out, axis=0)


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    lr: float,
    epochs: int,
    pos_weight: float,
    patience: int = 3,
) -> Dict[str, Any]:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        # val NLL
        model.eval()
        with torch.no_grad():
            vloss_sum = 0.0
            vn = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                vloss = criterion(logits, yb)
                vloss_sum += float(vloss.item()) * xb.size(0)
                vn += xb.size(0)
            vloss_mean = vloss_sum / max(vn, 1)

        # early stop
        if vloss_mean < best_val - 1e-4:
            best_val = vloss_mean
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_nll": best_val, "epochs_ran": ep}


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
    max_iter: int = 200,
) -> float:
    """
    Temperature scaling on validation logits.
    Returns fitted temperature (float).
    """
    scaler = TemperatureScaler(init_temp=1.0).to(device)
    logits_t = torch.from_numpy(logits.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad(set_to_none=True)
        scaled = scaler(logits_t)
        loss = F.binary_cross_entropy_with_logits(scaled, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(scaler.log_t).detach().cpu().item())
    return t


def ensemble_logits(models: List[nn.Module], loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mean_logits: float32 [N]
      var_logits:  float32 [N]  (across ensemble members)
    """
    all_logits = []
    for m in models:
        lg = predict_logits(m, loader, device)
        all_logits.append(lg.astype(np.float32))
    L = np.stack(all_logits, axis=0)  # [M,N]
    return L.mean(axis=0), L.var(axis=0)


def apply_policy_and_recall(
    pack: SelectorPack,
    rerank_mask: np.ndarray,
) -> float:
    """
    Apply policy: if rerank_mask[q]=True use full top1; else use global top1.
    Return R@1.
    """
    Nq = pack.Nq
    pred = np.where(rerank_mask, pack.top1_full, pack.topk_idx0[:, 0]).astype(np.int32)

    correct = top1_correct_from_pred(pack.gt_ptr, pack.gt_ind, pred)
    return float(correct.mean())


def sweep_thresholds(
    pack: SelectorPack,
    probs_rerank: np.ndarray,
    thresholds: np.ndarray,
    t0_ms: float = 0.0,
    t1_ms: float = 0.0,
    t2_ms: float = 0.0,
    has_depth: bool = False,
) -> List[Dict[str, Any]]:
    """
    Sweep thresholds for rerank decision.
    If has_depth is False, runtime is t = t0 + frac_rerank * (t1+t2) if you bundle reranks.
    """
    out = []
    for tau in thresholds:
        rerank_mask = probs_rerank < tau
        frac = float(rerank_mask.mean())
        r1 = apply_policy_and_recall(pack, rerank_mask)

        runtime = None
        if t0_ms > 0:
            extra = (t1_ms + t2_ms) if (t1_ms > 0 or t2_ms > 0) else 0.0
            runtime = t0_ms + frac * extra

        out.append({
            "tau": float(tau),
            "rerank_frac": frac,
            "R@1": r1,
            "est_ms_per_query": runtime,
        })
    return out


# -----------------------------
# CLI
# -----------------------------

def build_indices_from_split(split: np.ndarray, which: int) -> np.ndarray:
    return np.where(split == which)[0].astype(np.int64)

def _assert_csr_ok(gt_ptr: np.ndarray, gt_ind: np.ndarray):
    assert gt_ptr.ndim == 1 and gt_ind.ndim == 1
    assert gt_ptr[0] == 0
    assert gt_ptr[-1] == gt_ind.size, f"CSR mismatch: gt_ptr[-1]={gt_ptr[-1]} != len(gt_ind)={gt_ind.size}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train deep-ensemble selector on one dataset NPZ")
    t.add_argument("--npz", required=True, help="Training dataset NPZ containing q_split or use --split_json")
    t.add_argument("--out_dir", required=True)
    t.add_argument("--K", type=int, default=50, help="Use top-K similarities (will slice if file has larger K)")
    t.add_argument("--ens", type=int, default=5)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--subsample_train", type=int, default=1, help="Use every Nth training query for speed")
    t.add_argument("--seed", type=int, default=0)

    e = sub.add_parser("eval", help="Evaluate selector on a dataset NPZ (can be different dataset)")
    e.add_argument("--npz", required=True)
    e.add_argument("--ckpt_dir", required=True, help="Directory from train (contains ensemble checkpoints + temp.json)")
    e.add_argument("--K", type=int, default=50)
    e.add_argument("--batch", type=int, default=2048)
    e.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    e.add_argument("--t0_ms", type=float, default=0.0, help="global stage time per query (ms)")
    e.add_argument("--t_rerank_ms", type=float, default=0.0, help="rerank+depth time per query (ms)")

    args = ap.parse_args()

    os.makedirs(getattr(args, "out_dir", getattr(args, "ckpt_dir", ".")), exist_ok=True)

    if args.cmd == "train":
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        pack = load_pack(args.npz)

        # determine splits
        if pack.split is None:
            raise SystemExit("NPZ must include q_split (0=train,1=val,2=test) for train command.")

        # slice topK if needed
        K = min(args.K, pack.K)
        topk_val0 = pack.topk_val0[:, :K]
        feats = extract_features_from_topk(topk_val0)
        correct0, correct_full, y_benefit = make_benefit_labels(pack)

        # indices
        tr_idx = build_indices_from_split(pack.split, 0)
        va_idx = build_indices_from_split(pack.split, 1)

        if args.subsample_train > 1:
            tr_idx = tr_idx[::args.subsample_train]

        # class imbalance
        y_tr = y_benefit[tr_idx].astype(np.float32)
        pos = float(y_tr.sum())
        neg = float(y_tr.size - pos)
        pos_weight = (neg / max(pos, 1.0))
        print(f"[Data] train={len(tr_idx)} val={len(va_idx)}  pos={pos:.0f} neg={neg:.0f} pos_weight={pos_weight:.2f}")

        train_ds = TopKSelectorDataset(feats, y_benefit, tr_idx)
        val_ds = TopKSelectorDataset(feats, y_benefit, va_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

        d_in = feats.shape[1]
        models: List[MLPSelector] = []

        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Train ensemble
        for m in range(args.ens):
            seed = args.seed + 1000 * m
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = MLPSelector(d_in=d_in)
            stats = train_one(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=args.device,
                lr=args.lr,
                epochs=args.epochs,
                pos_weight=pos_weight,
                patience=3,
            )
            ckpt_path = os.path.join(out_dir, f"selector_m{m}.pt")
            torch.save({"state_dict": model.state_dict(), "d_in": d_in, "K": K, "seed": seed, "stats": stats}, ckpt_path)
            print(f"[Train] m={m} saved={ckpt_path} stats={stats}")
            models.append(model.to(args.device).eval())

        # Calibrate temperature on VAL ensemble mean logits
        val_loader_noshuf = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
        mean_logits, var_logits = ensemble_logits(models, val_loader_noshuf, args.device)
        y_val = y_benefit[va_idx].astype(np.float32)
        temp = fit_temperature(mean_logits, y_val, device=args.device)
        with open(os.path.join(out_dir, "temp.json"), "w") as f:
            json.dump({"temperature": temp, "label": "benefit", "K": K}, f, indent=2)
        print(f"[Calib] temperature={temp:.4f} saved={os.path.join(out_dir, 'temp.json')}")

        # Quick sweep on VAL: R@1 vs rerank%
        # Need the pack restricted to val queries for policy eval:
        # We'll hack by creating a view-pack with only val queries.
        # --- build a correct val pack (slice once; keep ptr/ind matched)
        gt_ptr_val, gt_ind_val = _slice_gt_ptr(pack.gt_ptr, pack.gt_ind, va_idx)
        _assert_csr_ok(gt_ptr_val, gt_ind_val)

        pack_val = SelectorPack(
            topk_idx0=pack.topk_idx0[va_idx, :K],
            topk_val0=pack.topk_val0[va_idx, :K],
            top1_full=pack.top1_full[va_idx],
            gt_ptr=gt_ptr_val,
            gt_ind=gt_ind_val,
            split=None,
        )

        r1_global_val = apply_policy_and_recall(pack_val, rerank_mask=np.zeros(pack_val.Nq, dtype=bool))
        r1_full_val   = apply_policy_and_recall(pack_val, rerank_mask=np.ones(pack_val.Nq, dtype=bool))
        print(f"[VAL sanity] R@1_global={r1_global_val:.4f}  R@1_full={r1_full_val:.4f}")

        print(f"[VAL sanity] top1_global min/max = {pack_val.topk_idx0[:,0].min()}/{pack_val.topk_idx0[:,0].max()}")
        print(f"[VAL sanity] top1_full   min/max = {pack_val.top1_full.min()}/{pack_val.top1_full.max()}")
        print(f"[VAL sanity] avg GT positives/query = {pack_val.gt_ind.size / max(pack_val.Nq,1):.3f}")

        probs = 1.0 / (1.0 + np.exp(-(mean_logits / temp)))  # sigmoid(scaled)
        print(f"[VAL probs] min={probs.min():.4f} p10={np.quantile(probs,0.1):.4f} "
        f"p50={np.quantile(probs,0.5):.4f} p90={np.quantile(probs,0.9):.4f} max={probs.max():.4f}")

        thresholds = np.linspace(0.0, 1.0, 401)
        rows = sweep_thresholds(pack_val, probs, thresholds)

        target = 0.935  # choose
        best = min((r for r in rows if r["R@1"] >= target), key=lambda r: r["rerank_frac"], default=None)
        print(f"[VAL target] R@1>={target} best={best}")
        rows = sweep_thresholds(pack_val, probs, thresholds)
        print("[VAL sweep] tau  rerank%   R@1")
        for r in rows[::2]:
            print(f"  {r['tau']:.2f}  {100*r['rerank_frac']:.1f}%  {r['R@1']:.4f}")

    elif args.cmd == "eval":
        pack = load_pack(args.npz)
        K = min(args.K, pack.K)

        # select split
        if args.split != "all":
            if pack.split is None:
                raise SystemExit("NPZ has no q_split; use --split all or add q_split.")
            which = {"train": 0, "val": 1, "test": 2}[args.split]
            idx = build_indices_from_split(pack.split, which)
        else:
            idx = np.arange(pack.Nq, dtype=np.int64)

        gt_ptr_s, gt_ind_s = _slice_gt_ptr(pack.gt_ptr, pack.gt_ind, idx)
        _assert_csr_ok(gt_ptr_s, gt_ind_s)

        pack_s = SelectorPack(
            topk_idx0=pack.topk_idx0[idx, :K],
            topk_val0=pack.topk_val0[idx, :K],
            top1_full=pack.top1_full[idx],
            gt_ptr=gt_ptr_s,
            gt_ind=gt_ind_s,
            split=None,
        )

        feats = extract_features_from_topk(pack_s.topk_val0)
        _, _, y_benefit = make_benefit_labels(pack_s)

        ds = TopKSelectorDataset(feats, y_benefit, np.arange(pack_s.Nq))
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

        # load ensemble
        models = []
        # temp
        temp_path = os.path.join(args.ckpt_dir, "temp.json")
        with open(temp_path, "r") as f:
            temp = float(json.load(f)["temperature"])

        # find checkpoints
        ckpts = sorted([p for p in os.listdir(args.ckpt_dir) if p.startswith("selector_m") and p.endswith(".pt")])
        if not ckpts:
            raise SystemExit("No selector_m*.pt in ckpt_dir")

        d_in = feats.shape[1]
        for c in ckpts:
            obj = torch.load(os.path.join(args.ckpt_dir, c), map_location="cpu")
            m = MLPSelector(d_in=d_in)
            m.load_state_dict(obj["state_dict"])
            m = m.to(args.device).eval()
            models.append(m)

        mean_logits, var_logits = ensemble_logits(models, loader, args.device)
        probs = 1.0 / (1.0 + np.exp(-(mean_logits / temp)))

        # baseline R@1
        r1_global = apply_policy_and_recall(pack_s, rerank_mask=np.zeros(pack_s.Nq, dtype=bool))
        r1_full = apply_policy_and_recall(pack_s, rerank_mask=np.ones(pack_s.Nq, dtype=bool))

        # sweep
        thresholds = np.linspace(0.0, 1.0, 41)
        rows = sweep_thresholds(
            pack_s,
            probs_rerank=probs,
            thresholds=thresholds,
            t0_ms=args.t0_ms,
            t1_ms=args.t_rerank_ms,
            t2_ms=0.0,
            has_depth=False,
        )

        print(f"[Eval] split={args.split} N={pack_s.Nq}  R@1_global={r1_global:.4f}  R@1_full={r1_full:.4f}")
        print("tau   rerank%   R@1    est_ms/q")
        for r in rows[::4]:
            ms = r["est_ms_per_query"]
            ms_s = "-" if ms is None else f"{ms:.2f}"
            print(f"{r['tau']:.2f}  {100*r['rerank_frac']:.1f}%  {r['R@1']:.4f}  {ms_s}")

        # dump full sweep json for plotting
        out_json = os.path.join(args.ckpt_dir, f"sweep_{os.path.basename(args.npz).replace('.npz','')}_{args.split}.json")
        with open(out_json, "w") as f:
            json.dump(
                {
                    "npz": args.npz,
                    "split": args.split,
                    "K": K,
                    "temperature": temp,
                    "R@1_global": r1_global,
                    "R@1_full": r1_full,
                    "rows": rows,
                },
                f,
                indent=2,
            )
        print(f"[Saved] {out_json}")


def _slice_gt_ptr(gt_ptr: np.ndarray, gt_ind: np.ndarray, q_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice CSR rows to a subset of queries q_idx, returning new (ptr, ind).
    This makes evaluation on a split easy without carrying full GT.
    """
    q_idx = q_idx.astype(np.int64)
    # compute nnz per selected row
    counts = (gt_ptr[q_idx + 1] - gt_ptr[q_idx]).astype(np.int64)
    new_ptr = np.zeros(q_idx.size + 1, dtype=np.int64)
    new_ptr[1:] = np.cumsum(counts)

    new_ind = np.empty(int(new_ptr[-1]), dtype=np.int32)
    w = 0
    for i, q in enumerate(q_idx):
        a = int(gt_ptr[q]); b = int(gt_ptr[q + 1])
        n = b - a
        new_ind[w:w+n] = gt_ind[a:b]
        w += n

    # Ensure each row sorted for fast membership checks
    # If you know your GT is already sorted, you can remove this.
    w = 0
    for i in range(q_idx.size):
        a = int(new_ptr[i]); b = int(new_ptr[i+1])
        new_ind[a:b].sort()

    return new_ptr, new_ind


if __name__ == "__main__":
    main()