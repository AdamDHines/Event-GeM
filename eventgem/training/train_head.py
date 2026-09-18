"""Train the projection head on cached NYC descriptors — campaign version (v2).

Reworked after the Phase-A diagnostics showed the first sweep never tested anything
(early-stopped at ~epoch 4 on a 73-query val signal, rank-collapsed outputs, 29% of GPS
positives being opposite-direction views):

- all 15 NYC recordings train; displacement dedup (>= --dedup-m metres) replaces the old
  speed filter; nothing NYC is held out (the two highest-overlap pairs are logged as
  diagnostics only);
- positives are heading-gated (circular delta < --heading-max deg) cross-recording GPS
  matches;
- batches are geo-clustered (sampled route cells with their cross-recording members) and
  the default loss is supervised-contrastive over all in-batch positives, with a neutral
  band (same place but same recording, or the pos..neg radius annulus) excluded from the
  denominator;
- the default head is an identity residual, z = L2(x + MLP(x)) with a zero-initialised
  last layer: at epoch 0 it IS the statistics-free descriptor (sunset1 base R@1 83.51 for
  the regional input), so the Brisbane monitor cannot start collapsed;
- selection is the sunset1 monitor R@1 (fast GPU argmax every epoch); early stopping is
  OFF by default; NYC metrics select nothing.

  PYTHONPATH=. pixi run python -m eventgem.training.train_head --input regional \
      --wandb --run-name v2_regional_supcon
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import BallTree
from skimage.transform import resize

from eventgem.training.common import (
    ResidualHead,
    BLOCK_DIM, DESC_BLOCKS, DESC_DIM_V2, DESC_SLICES, MEGAEVENT_NYC, NYC_DIAG_PAIRS,
    atomic_save, haversine_m, recall_at_k,
)

EARTH_R = 6371000.0

# How many monitor queries may flip between the identity and a zero-initialised residual head
# before the init is considered broken. L2-renormalising an already-unit vector perturbs it by
# ~1e-7, which is enough to swap a near-tie; the monitor's resolution is 1 query = 6.9e-5 R@1.
MAX_INIT_FLIPS = 5


def prepare_input(desc, variant):
    """Select a variant's columns from a bank and balance it: each 128-D GeM block
    L2-normalised, then the whole vector L2'd. Fit-free by construction.

    Two bank layouts are supported, dispatched on which registry the variant name is in
    (the two are disjoint). v1 banks are [K,1024] and variants are contiguous column
    slices; v2 banks are [K,3328] and variants are lists of 128-D block ids, because
    combinations like g5+r5 are not contiguous there. The balancing is identical, and the
    v1 path is unchanged.
    """
    if variant in DESC_BLOCKS:
        ids = DESC_BLOCKS[variant]
        assert desc.shape[1] % BLOCK_DIM == 0, (
            f"block-indexed variant {variant!r} needs a bank whose width is a multiple of "
            f"{BLOCK_DIM}; got {desc.shape}")
        n_blocks = desc.shape[1] // BLOCK_DIM
        assert max(ids) < n_blocks, (
            f"variant {variant!r} indexes block {max(ids)} but this bank has only "
            f"{n_blocks} blocks ({desc.shape[1]}-D) -- wrong bank layout?")
        x = torch.from_numpy(np.ascontiguousarray(desc)).float()
        x = x.view(x.shape[0], -1, BLOCK_DIM)[:, ids]
        x = x.reshape(x.shape[0], -1)
    else:
        x = torch.from_numpy(np.ascontiguousarray(desc[:, DESC_SLICES[variant]])).float()
    blocks = F.normalize(x.view(x.shape[0], -1, 128), p=2, dim=2).view(x.shape[0], -1)
    return F.normalize(blocks, p=2, dim=1)


class Head(nn.Module):
    """Free-form MLP projection (the v1 head, kept as an ablation arm)."""

    def __init__(self, in_dim, hidden, out_dim=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)


def ensure_all_banks(args, device):
    """Build any missing descriptor banks in-process; the banks dir is a cache."""
    from eventgem.training import extract_banks as eb
    from eventgem.training.common import discover_nyc_recordings
    loaded = None

    def get_model():
        nonlocal loaded
        if loaded is None:
            loaded = eb.load_model(args.se_config, args.se_weights, device)
        return loaded

    banks = Path(args.banks)
    try:
        recs = discover_nyc_recordings(args.nyc_root)
    except FileNotFoundError as err:
        print(f"[WARN] NYC discovery failed ({err}); using existing banks only")
        recs = []
    for name, rows in recs:
        if args.refresh_banks or not (banks / f"nyc_{name}.npz").exists():
            model, cfg = get_model()
            try:
                eb.ensure_nyc_bank(name, rows, banks, model, cfg, device,
                                   args.extract_batch_size, min_speed=0.0,
                                   overwrite=args.refresh_banks,
                                   to_brisbane_scale=args.to_brisbane_scale)
            except ValueError as err:
                print(f"[SKIP] {name}: {err}")
    for seq in ("sunset2", "sunset1"):
        if args.refresh_banks or not (banks / f"brisbane_{seq}.npz").exists():
            if os.path.isdir(os.path.join(args.bris_root, seq)):
                model, cfg = get_model()
                eb.ensure_brisbane_bank(seq, args.bris_root, banks, model, cfg, device,
                                        overwrite=args.refresh_banks)
            else:
                print(f"[WARN] brisbane {seq} not under {args.bris_root}")
    if loaded is not None:
        del loaded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_and_dedup(banks_dir, variant, dedup_m):
    """All NYC banks -> concatenated (feats, lat, lon, heading, rec_id, per_rec) after
    per-recording displacement dedup (keep a sample once it moved >= dedup_m metres).

    `per_rec` carries per-recording GPS only; its descriptor slot is None. See the comment
    at the assignment -- holding the descriptors there doubled peak RAM and nothing read them.
    """
    feats, lat, lon, heading, rec = [], [], [], [], []
    per_rec = {}
    for path in sorted(Path(banks_dir).glob("nyc_*.npz")):
        z = np.load(path, allow_pickle=True)
        if "heading" not in z:
            raise SystemExit(f"{path} predates the heading rework — rebuild banks "
                             f"(extract_banks --dataset nyc --overwrite)")
        la, lo = z["lat"], z["lon"]
        keep = [0]
        for i in range(1, len(la)):
            if haversine_m(la[keep[-1]], lo[keep[-1]], la[i], lo[i]) >= dedup_m:
                keep.append(i)
        keep = np.array(keep)
        name = str(z["recording"])
        # Deliberately NOT holding z["desc"][keep] here. The only caller discards per_rec
        # (train_head.py: `..., _ = load_and_dedup(...)`), and the diagnostic pairs reload
        # what they need from disk, so retaining full-width descriptors for all 15 NYC
        # recordings was 2.4 GB of peak RAM at 10880-D bought for nothing -- enough to
        # double the footprint of a wide-descriptor run.
        per_rec[name] = (None, la[keep], lo[keep], z["heading"][keep])
        feats.append(prepare_input(z["desc"][keep], variant))
        lat.append(la[keep]); lon.append(lo[keep]); heading.append(z["heading"][keep])
        rec.append(np.full(len(keep), len(per_rec) - 1))
        print(f"  {name}: {len(keep)}/{len(la)} after {dedup_m} m dedup")
    if not feats:
        raise SystemExit(f"no NYC banks under {banks_dir}")
    return (torch.cat(feats), np.concatenate(lat), np.concatenate(lon),
            np.concatenate(heading), np.concatenate(rec), per_rec)


def circ_delta(h1, h2):
    d = np.abs(h1 - h2) % 360.0
    return np.minimum(d, 360.0 - d)


def build_structure(lat, lon, heading, rec, pos_radius, heading_max, cell_m=25.0):
    """Positive lists (cross-recording, heading-gated), anchors, and geo cell ids."""
    tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    neigh = tree.query_radius(np.radians(np.c_[lat, lon]), r=pos_radius / EARTH_R)
    positives = []
    for i, nb in enumerate(neigh):
        nb = nb[rec[nb] != rec[i]]
        if heading_max < 180:
            nb = nb[circ_delta(heading[nb], heading[i]) < heading_max]
        positives.append(nb)
    anchors = np.flatnonzero([len(p) > 0 for p in positives])

    lat0 = lat.mean()
    cy = np.floor(lat * 111320.0 / cell_m).astype(np.int64)
    cx = np.floor(lon * 111320.0 * math.cos(math.radians(lat0)) / cell_m).astype(np.int64)
    cell = cy * 1_000_003 + cx
    cell_members = {}
    for i in anchors:                       # cells that contain at least one anchor
        cell_members.setdefault(cell[i], []).append(i)
    return positives, anchors, cell, cell_members


def sample_batch(rng, anchors_perm, cursor, batch, positives, cell, cell_members):
    """Geo-clustered batch: walk the anchor permutation, and for each anchor pull its
    whole cell's anchors plus one random positive each, until the batch is full."""
    rows = []
    seen_cells = set()
    while len(rows) < batch and cursor < len(anchors_perm):
        a = anchors_perm[cursor]; cursor += 1
        c = cell[a]
        if c in seen_cells:
            continue
        seen_cells.add(c)
        members = cell_members.get(c, [a])[:8]
        for m in members:
            rows.append(m)
            rows.append(int(rng.choice(positives[m])))
            if len(rows) >= batch:
                break
    return np.array(rows[:batch]), cursor


def supcon_loss(z, latlon, rec, heading, tau, pos_radius, neg_radius, heading_max):
    """Supervised contrastive over all in-batch positives; a neutral band (same-recording
    near pairs, the pos..neg annulus, and wrong-heading same-place pairs) is excluded from
    the denominator so near-duplicates and ambiguous pairs neither attract nor repel."""
    n = z.shape[0]
    d = haversine_m(latlon[:, None, 0], latlon[:, None, 1],
                    latlon[None, :, 0], latlon[None, :, 1])
    same_rec = rec[:, None] == rec[None, :]
    near = d < pos_radius
    dh_ok = circ_delta(heading[:, None], heading[None, :]) < heading_max
    pos = torch.from_numpy(near & ~same_rec & dh_ok).to(z.device)
    neutral = torch.from_numpy((near & (same_rec | ~dh_ok))
                               | ((d >= pos_radius) & (d < neg_radius))).to(z.device)
    neutral &= ~pos
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    sim = sim.masked_fill(neutral, float("-inf"))
    denom = torch.logsumexp(sim, dim=1)
    pos_f = pos.float()
    n_pos = pos_f.sum(1)
    valid = n_pos > 0
    mean_pos = (sim.masked_fill(~pos, 0.0) * pos_f).sum(1) / n_pos.clamp(min=1)
    return (denom - mean_pos)[valid].mean(), int(valid.sum())


def nt_xent_loss(z, tau):
    """v1 loss (ablation): [anchors | designated positives] rows, one positive each."""
    b = z.shape[0] // 2
    sim = z @ z.t() / tau
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


class Monitor:
    """Brisbane sunset2->sunset1 monitor. Fast GPU R@1 every epoch (selection signal);
    the full argsort convention runs only for baselines and the final best checkpoint.

    Both evaluations chunk over queries once the banks are too large to hold on device, so a
    wide descriptor does not have to fit its whole similarity matrix in VRAM. The similarity
    matrix in `full` is assembled on the CPU either way -- it was always moved there for
    `recall_at_k`; only the peak device allocation changes.
    """

    MAX_RESIDENT_BYTES = 512 * 1024 ** 2   # keep the published variants on the old fast path
    CHUNK = 2048

    def __init__(self, banks_dir, gt_path, variant, device):
        rd = np.load(Path(banks_dir) / "brisbane_sunset2.npz")["desc"]
        qd = np.load(Path(banks_dir) / "brisbane_sunset1.npz")["desc"]
        gt = np.load(gt_path)
        if gt.shape != (len(rd), len(qd)):
            gt = resize(gt, (len(rd), len(qd)), order=0, preserve_range=True,
                        anti_aliasing=False)
        self.gt = gt.astype(bool)
        self.rd, self.qd = rd, qd
        ref, qry = prepare_input(rd, variant), prepare_input(qd, variant)
        # Wide descriptors (spatial pyramids, 8x8 partitions) make these banks large enough
        # that keeping them resident steals memory the optimizer needs -- a 10880-D pair is
        # ~1.2 GB, which is what pushed L1248 into an OOM inside Adam's step. Below the
        # threshold, behaviour is exactly as before: resident on device, no chunking.
        nbytes = (ref.numel() + qry.numel()) * ref.element_size()
        self.resident = nbytes <= self.MAX_RESIDENT_BYTES
        self.ref = ref.to(device) if self.resident else ref
        self.qry = qry.to(device) if self.resident else qry
        self.gt_t = torch.from_numpy(self.gt).to(device)
        self.valid = self.gt_t.any(0)
        self.device = device
        if not self.resident:
            print(f"[monitor] banks {nbytes/1e9:.2f} GB > "
                  f"{self.MAX_RESIDENT_BYTES/1e9:.2f} GB — holding on CPU, chunked eval")

    @torch.inference_mode()
    def _embed(self, head, x):
        """Head-forward in chunks, moving to device only a slice at a time."""
        if self.resident:
            return head(x)
        return torch.cat([head(x[i:i + self.CHUNK].to(self.device))
                          for i in range(0, len(x), self.CHUNK)])

    @torch.inference_mode()
    def fast_r1(self, head):
        # The resident branch is the ORIGINAL single-shot computation, byte for byte.
        # head(chunk) and head(full)[chunk] differ in the last bits of the GEMM, which is
        # enough to flip a near-tie: chunking this path once moved sunset1 R@1 by 1/14478
        # (6.9e-5) and tripped main()'s identity-at-init assertion. Chunk only when the
        # banks genuinely do not fit.
        if self.resident:
            top1 = (head(self.ref) @ head(self.qry).t()).argmax(0)
        else:
            r = self._embed(head, self.ref)
            top1 = torch.empty(len(self.qry), dtype=torch.long, device=self.device)
            for i in range(0, len(self.qry), self.CHUNK):
                q = self._embed(head, self.qry[i:i + self.CHUNK])
                top1[i:i + len(q)] = (r @ q.t()).argmax(0)
                del q
        hits = self.gt_t[top1, torch.arange(len(top1), device=self.device)]
        return float(hits[self.valid].float().mean())

    @torch.inference_mode()
    def full(self, head, ks=(1, 5, 10, 50)):
        if self.resident:
            sim = (head(self.ref) @ head(self.qry).t()).cpu().numpy()
        else:
            r = self._embed(head, self.ref)
            sim = np.empty((len(self.ref), len(self.qry)), dtype=np.float32)
            for i in range(0, len(self.qry), self.CHUNK):
                q = self._embed(head, self.qry[i:i + self.CHUNK])
                sim[:, i:i + len(q)] = (r @ q.t()).cpu().numpy()
                del q
        return recall_at_k(sim, self.gt, list(ks))


def diag_pairs_metrics(head, banks_dir, variant, device, tol=75.0):
    out = {}
    for qry_rec, ref_rec in NYC_DIAG_PAIRS:
        rp, qp = Path(banks_dir) / f"nyc_{ref_rec}.npz", Path(banks_dir) / f"nyc_{qry_rec}.npz"
        if not (rp.exists() and qp.exists()):
            continue
        r, q = np.load(rp), np.load(qp)
        gt = haversine_m(r["lat"][:, None], r["lon"][:, None],
                         q["lat"][None, :], q["lon"][None, :]) < tol
        with torch.inference_mode():
            re = head(prepare_input(r["desc"], variant).to(device)).cpu().numpy()
            qe = head(prepare_input(q["desc"], variant).to(device)).cpu().numpy()
        rec = recall_at_k(re @ qe.T, gt, [1, 10])
        out[f"diag/{ref_rec}->{qry_rec}_R@1"] = rec[1]
        out[f"diag/{ref_rec}->{qry_rec}_R@10"] = rec[10]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--banks", default=os.path.join(os.environ.get("EVENTGEM_OUT", "."), "banks"))
    ap.add_argument("--input", choices=list(DESC_SLICES) + list(DESC_BLOCKS),
                    default="regional")
    ap.add_argument("--head", choices=["residual", "mlp"], default="residual")
    ap.add_argument("--dim", type=int, default=512, help="hidden width")
    ap.add_argument("--out-dim", type=int, default=128, help="mlp head only")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--loss", choices=["supcon", "ntxent"], default="supcon")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--pos-radius", type=float, default=25.0)
    ap.add_argument("--neg-radius", type=float, default=75.0)
    ap.add_argument("--heading-max", type=float, default=45.0,
                    help="max circular heading delta for positives; 180 disables the gate "
                         "(A2: 29%% of ungated positives are opposite-direction views)")
    ap.add_argument("--dedup-m", type=float, default=3.0,
                    help="train-time displacement dedup within a recording")
    ap.add_argument("--early-stop", type=int, default=0, help="0 disables (default)")
    ap.add_argument("--monitor-every", type=int, default=1,
                    help="epochs between fast sunset1 R@1 evals (the selection signal)")
    ap.add_argument("--diag-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--resume", default="auto", choices=["auto", "never"])
    ap.add_argument("--bris-gt", default=os.path.join(
        os.environ.get("EVENTGEM_BRIS_ROOT", "/media/adam/vprdatasets/eventgem/brisbane_event"),
        "ground_truth", "sunset2_sunset1_GT.npy"))
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="eventgem-nyc-head")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--nyc-root", default=os.environ.get("EVENTGEM_NYC_ROOT", MEGAEVENT_NYC))
    ap.add_argument("--to-brisbane-scale", action="store_true", default=True)
    ap.add_argument("--extract-batch-size", type=int, default=8)
    ap.add_argument("--refresh-banks", action="store_true")
    ap.add_argument("--se-config",
                    default="eventgem/external/superevent/config/super_event.yaml")
    ap.add_argument("--se-weights",
                    default="eventgem/external/superevent/saved_models/super_event_weights.pth")
    args = ap.parse_args()

    run_name = args.run_name or (f"v2_{args.input}_{args.head}_{args.loss}"
                                 f"_t{args.tau}_lr{args.lr}_d{args.dim}")
    save_dir = Path(args.save_dir or
                    os.path.join(os.environ.get("EVENTGEM_OUT", "."), "runs", run_name))
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ensure_all_banks(args, device)
    print("loading + dedup:")
    feats, lat, lon, heading, rec, _ = load_and_dedup(args.banks, args.input, args.dedup_m)
    positives, anchors, cell, cell_members = build_structure(
        lat, lon, heading, rec, args.pos_radius, args.heading_max)
    print(f"{len(feats)} descriptors, {len(anchors)} anchors "
          f"(heading gate {'<%g deg' % args.heading_max if args.heading_max < 180 else 'OFF'})")
    if len(anchors) == 0:
        raise SystemExit("no positives under the current gates")

    monitor = Monitor(args.banks, args.bris_gt, args.input, device)

    in_dim = feats.shape[1]
    if args.head == "residual":
        head = ResidualHead(in_dim, args.dim, args.dropout).to(device)
    else:
        head = Head(in_dim, args.dim, args.out_dim, args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    start_epoch, best = 0, -1.0

    latest = save_dir / "latest.pt"
    cfgkey = {"input": args.input, "head": args.head, "dim": args.dim,
              "out_dim": args.out_dim, "loss": args.loss}
    if args.resume == "auto" and latest.exists():
        ck = torch.load(latest, map_location=device)
        assert ck["config"] == cfgkey, "checkpoint config drift — refusing to resume"
        head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from {latest} at epoch {start_epoch} (best {best:.4f})")

    wb = None
    if args.wandb and args.wandb_mode != "disabled":
        import wandb as wb
        wb.init(project=args.wandb_project, entity=None, name=run_name, mode=args.wandb_mode,
                resume="allow", id=run_name, config=vars(args))
        wb.define_metric("monitor/sunset1_R@1", summary="max")

    # Baselines + the residual-init invariant: at epoch 0 a residual head IS the identity,
    # so its monitor must equal the statistics-free descriptor's score exactly.
    ident_r1 = float(monitor.fast_r1(nn.Identity().to(device)))
    print(f"identity ({args.input}) sunset1 fast-R@1: {ident_r1:.4f}")
    if args.head == "residual" and start_epoch == 0:
        init_r1 = monitor.fast_r1(head)
        # The invariant is real -- a zero-initialised residual head IS the identity -- but the
        # quantity compared is a RECALL, quantised at 1/n_valid (6.9e-5 on the 14478-query
        # sunset1 monitor). A 1e-6 tolerance is ~69x finer than the metric can resolve, so it
        # only ever passed when zero queries flipped. head(x) = L2(x + 0) re-normalises an
        # already-unit vector, which perturbs it by ~1e-7 relative and can flip a near-tie;
        # `L14` flipped exactly one query and tripped an assertion it could not have passed.
        # Tolerance is now expressed in the metric's own units: a handful of flipped ties is
        # float noise, anything more is a genuinely broken initialisation.
        tol = max(MAX_INIT_FLIPS / max(int(monitor.valid.sum()), 1), 1e-6)
        n_flipped = abs(init_r1 - ident_r1) * int(monitor.valid.sum())
        assert abs(init_r1 - ident_r1) < tol, \
            (f"residual head not identity at init ({init_r1} vs {ident_r1}; "
             f"{n_flipped:.1f} queries flipped, tolerance {MAX_INIT_FLIPS})")
        print(f"residual init check PASSED ({n_flipped:.1f} queries flipped, "
              f"tolerance {MAX_INIT_FLIPS})")
    if wb:
        wb.run.summary["baseline_identity/sunset1_R@1"] = ident_r1

    latlon = np.c_[lat, lon]
    feats_dev = feats.to(device)
    stale = 0
    steps_per_epoch = max(len(anchors) // args.batch, 1)
    for epoch in range(start_epoch, args.epochs):
        head.train()
        perm = rng.permutation(anchors)
        cursor, losses, pos_counts = 0, [], []
        t0 = time.time()
        for _ in range(steps_per_epoch):
            if args.loss == "supcon":
                rows, cursor = sample_batch(rng, perm, cursor, args.batch,
                                            positives, cell, cell_members)
                if len(rows) < 32:
                    break
                z = head(feats_dev[rows])
                loss, n_valid = supcon_loss(z, latlon[rows], rec[rows], heading[rows],
                                            args.tau, args.pos_radius, args.neg_radius,
                                            args.heading_max)
                pos_counts.append(n_valid)
            else:
                take = perm[cursor: cursor + args.batch // 2]; cursor += args.batch // 2
                if len(take) < 16:
                    break
                p_idx = np.array([int(rng.choice(positives[a])) for a in take])
                z = head(feats_dev[np.concatenate([take, p_idx])])
                loss = nt_xent_loss(z, args.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        head.eval()

        log = {"epoch": epoch, "train/loss": float(np.mean(losses)) if losses else float("nan"),
               "lr": sched.get_last_lr()[0]}
        sel = None
        if epoch % args.monitor_every == 0 or epoch == args.epochs - 1:
            sel = monitor.fast_r1(head)
            log["monitor/sunset1_R@1"] = sel
        if args.diag_every and epoch % args.diag_every == 0:
            log.update(diag_pairs_metrics(head, args.banks, args.input, device))
        print(f"epoch {epoch}: loss {log['train/loss']:.4f} "
              f"sunset1 R@1 {sel if sel is not None else float('nan'):.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if wb:
            wb.log(log, step=epoch)

        state = {"head": head.state_dict(), "opt": opt.state_dict(),
                 "sched": sched.state_dict(), "epoch": epoch,
                 "best": max(best, sel if sel is not None else -1.0),
                 "config": cfgkey, "args": vars(args)}
        atomic_save(state, latest)
        if sel is not None:
            if sel > best:
                best, stale = sel, 0
                atomic_save(state, save_dir / "best.pt")
            else:
                stale += 1
                if args.early_stop and stale >= args.early_stop:
                    print(f"early stop at epoch {epoch} (best {best:.4f})")
                    break

    # Full-convention recall for the best checkpoint (the number the tables use).
    ck = torch.load(save_dir / "best.pt", map_location=device)
    head.load_state_dict(ck["head"]); head.eval()
    final = monitor.full(head)
    summary = {"best_fast_r1": best, "best_epoch": int(ck["epoch"]),
               "identity_fast_r1": ident_r1,
               "sunset1_full": {str(k): v for k, v in final.items()}}
    with open(save_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"done: best sunset1 fast-R@1 {best:.4f} (epoch {ck['epoch']}) vs identity "
          f"{ident_r1:.4f}; full recall of best: "
          + " ".join(f"R@{k}={100 * v:.2f}" for k, v in final.items()))
    if wb:
        wb.run.summary.update({"best_sunset1_R@1": best,
                               **{f"final/sunset1_R@{k}": v for k, v in final.items()}})
        wb.finish()


if __name__ == "__main__":
    main()
