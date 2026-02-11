'''
Imports
'''
import os
import sys
import torch
import yaml
import math
import requests

import numpy as np
import torch.nn.functional as F
import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from eventgem.utils.generate_mcts import gen_mcts
from eventgem.dataset import EventGeMData, EventGeMMCTS
from eventgem.utils.eventlab_config import update_config
from eventgem.utils.ckpt_downloader import download_backbone_ckpt
from eventgem.utils.rerank_utils import load_event_features, process_single_query

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Backbone path (already working)
BACKBONE_ROOT = os.path.join(THIS_DIR, "external", "backbone")

# Superevent root — where the "models" package lives
SUPEREVENT_ROOT = os.path.join(THIS_DIR, "external", "superevent")

for path in (BACKBONE_ROOT, SUPEREVENT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Backbone import (unchanged)
from eventgem.external.backbone.model.ours_model.ours_model_pretrain import vit_contrastive_patch16_small

# Now "models" resolves to eventgem/external/superevent/models
from models.super_event import SuperEvent, SuperEventFullRes
from models.util import fast_nms



THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKBONE_ROOT = os.path.join(THIS_DIR, "external", "backbone")
if BACKBONE_ROOT not in sys.path:
    sys.path.insert(0, BACKBONE_ROOT)
from eventgem.external.backbone.model.ours_model.ours_model_pretrain import vit_contrastive_patch16_small

class EventGeM:
    def __init__(self, args):
        # Before running, ensure repository was cloned with --recurse-submodules
        submodule_paths = ["./eventgem/external/backbone", "./eventgem/external/eventlab"]
        for path in submodule_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Submodule path '{path}' not found. Please clone Event-GeM with --recurse-submodules.")

        # Set all args as class attributes
        for k in vars(args): 
            setattr(self, k, getattr(args, k))

        # Get and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def GeM(self, feats, p=5):
        return F.avg_pool2d((feats.clamp(min=1e-6)).pow(p), (feats.shape[-2], feats.shape[-1])).pow(1.0/p)

    def extract_features(self):
        # Define backbone model (This is a ViT, as confirmed by your checkpoint)
        backbone = vit_contrastive_patch16_small(mask_ratio=0.0, in_chans=2, num_classes=512)
        
        # Ensure the backbone checkpoint exists
        if not os.path.exists(self.backbone_ckpt):
            # ... (your download logic) ...
            pass
            
        print(f"Loading checkpoint from: {self.backbone_ckpt}")
        checkpoint = torch.load(self.backbone_ckpt, map_location='cpu')

        # --- FIX 1: UNWRAP THE NESTED DICTIONARY ---
        # The weights are hidden inside the "checkpoint" key
        if isinstance(checkpoint, dict) and "checkpoint" in checkpoint:
            state_dict = checkpoint["checkpoint"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # --- FIX 2: CLEAN PREFIXES ---
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove "encoder_q." (seen in your debug output) and "module."
            new_key = k.replace("encoder_q.", "").replace("module.", "")
            new_state_dict[new_key] = v

        # --- FIX 3: LOAD AND VERIFY ---
        # strict=False is required because the checkpoint contains extra keys 
        # (momentum encoder 'encoder_k', queues, etc.) that we don't need.
        msg = backbone.load_state_dict(new_state_dict, strict=False)
        
        # Validation: We expect 'missing_keys' to mostly be the head (event_head/image_head).
        # We expect 'unexpected_keys' to be the momentum encoder stuff.
        # CRITICAL: If 'patch_embed.proj.weight' is missing, the load actually failed.
        if "patch_embed.proj.weight" in msg.missing_keys:
             raise RuntimeError("CRITICAL FAILURE: The input layer (patch_embed) did not load!")
        
        print("Backbone loaded successfully.")
        backbone.to(self.device).eval()

        # Define the DataLoaders
        ref_dataset = EventGeMData(self.reference_path)
        query_dataset = EventGeMData(self.query_path)
        ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=self.backbone_batch_size, shuffle=False, num_workers=4)
        query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=self.backbone_batch_size, shuffle=False, num_workers=4)

        # Run feature extraction for reference and query sets
        outdir = os.path.join(self.feature_out, self.dataset, f"{self.reference}-{self.query}")
        os.makedirs(outdir, exist_ok=True)

        # Reference events
        ref_feats = []
        for events in tqdm(ref_loader, desc="Extracting reference features", unit="batch"):
            events = events.to(self.device)
            events = backbone.patch_embed(events)
            events = events + backbone.pos_embed
            cls_tokens = backbone.tokens.expand(events.shape[0], -1, -1)
            events = torch.cat((cls_tokens, events), dim=1)
            for blk in backbone.blocks:
                events = blk(events)
            events = backbone.norm(events)
            patch_tokens = events[:, 2:, :]
            B, N, C = patch_tokens.shape
            H = W = int(N**0.5)
            patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            # Perform GeM pooling
            feats = self.GeM(patch_tokens)
            ref_feats.append(feats.squeeze(-1).squeeze(-1).detach().cpu())
        ref_feats = torch.cat(ref_feats, dim=0)
        torch.save(ref_feats, os.path.join(outdir, f"{self.dataset}_{self.reference}_features.pt"))

        # Query events
        query_feats = []
        for events in tqdm(query_loader, desc="Extracting query features", unit="batch"):
            events = events.to(self.device)
            events = backbone.patch_embed(events)
            events = events + backbone.pos_embed
            cls_tokens = backbone.tokens.expand(events.shape[0], -1, -1)
            events = torch.cat((cls_tokens, events), dim=1)
            for blk in backbone.blocks:
                events = blk(events)
            events = backbone.norm(events)
            patch_tokens = events[:, 2:, :]
            B, N, C = patch_tokens.shape
            H = W = int(N**0.5)
            patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
            # Perform GeM pooling
            feats = self.GeM(patch_tokens)
            query_feats.append(feats.squeeze(-1).squeeze(-1).detach().cpu())
        query_feats = torch.cat(query_feats, dim=0)
        torch.save(query_feats, os.path.join(outdir, f"{self.dataset}_{self.query}_features.pt"))

        # Compute cosine similarity
        ref_feats = F.normalize(ref_feats, p=2, dim=1)
        query_feats = F.normalize(query_feats, p=2, dim=1)
        sim_matrix = torch.matmul(query_feats, ref_feats.t()).T
        torch.save(sim_matrix, os.path.join(outdir, f"{self.dataset}_{self.reference}_{self.query}_similarity.pt"))

    def feature_inference(self):
        # Check that the specified datasets exist - need frame reconstructued directories
        root = self.data_root
        self.reference_path = os.path.join(root, self.dataset, self.reference, f"{self.reference}-frames-{self.recon_msec}")
        self.query_path = os.path.join(root, self.dataset, self.query, f"{self.query}-frames-{self.recon_msec}")
        update_config(root, self.dataset, self.reference, self.query, time=self.recon_msec)

        self.extract_features()

    # ----------------------------------------------------------
    # SuperEvent / MCTS helpers
    # ----------------------------------------------------------

    def load_superevent_config(self, config_path: Path):
        """Load main config and merge with backbone config if specified."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "backbone" in config:
            # Try a couple of relative locations for the backbone config
            backbone_config_path = config_path.parent.parent / "backbones" / f"{config['backbone']}.yaml"
            if not backbone_config_path.exists():
                backbone_config_path = config_path.parent / "backbones" / f"{config['backbone']}.yaml"

            if backbone_config_path.exists():
                with open(backbone_config_path, "r") as f:
                    backbone_cfg = yaml.safe_load(f)
                # Python 3.9-safe dict merge
                config.update(backbone_cfg)
                config["backbone_config"]["input_channels"] = config["input_channels"]
            else:
                print(f"[WARN] Could not find backbone config for {config['backbone']}")

        return config


    def build_superevent_model(self, config_path: Path, weights_path: Path, device: torch.device):
        """Initialize and load weights for the SuperEvent model."""
        config = self.load_superevent_config(config_path)

        if config.get("pixel_wise_predictions", False):
            model = SuperEventFullRes(config, tracing=False)
        else:
            model = SuperEvent(config, tracing=False)

        print(f"[INFO] Loading SuperEvent weights from {weights_path}")
        state = torch.load(weights_path, map_location=device)
        # If checkpoint is a dict, extract actual state dict
        if isinstance(state, dict) and any(k in state for k in ("model", "state_dict")):
            state = state.get("model", state.get("state_dict", state))

        model.load_state_dict(state)
        model.to(device).eval()
        return model, config


    def sample_descriptors_at_kpts(self, keypoints, descriptors, Hc, Wc):
        """
        Sample descriptors at keypoint locations using bilinear interpolation
        on the descriptor map.

        keypoints: (N, 2) in (y, x)
        descriptors: (1, C, Hc, Wc)
        """
        # (y, x) -> (x, y)
        kpts_xy = keypoints.float()[:, [1, 0]]

        grid = torch.zeros(
            (1, 1, kpts_xy.shape[0], 2),
            dtype=kpts_xy.dtype,
            device=kpts_xy.device,
        )
        # x-coordinate (width)
        grid[0, 0, :, 0] = 2.0 * kpts_xy[:, 0] / (Wc - 1) - 1.0
        # y-coordinate (height)
        grid[0, 0, :, 1] = 2.0 * kpts_xy[:, 1] / (Hc - 1) - 1.0

        # (1, C, Hc, Wc) + (1, 1, N, 2) -> (1, C, 1, N)
        desc_sampled = F.grid_sample(
            descriptors, grid, mode="bilinear", align_corners=True
        )
        # -> (N, C)
        desc_sampled = desc_sampled[0, :, 0, :].t()
        desc_sampled = F.normalize(desc_sampled, p=2, dim=1)
        return desc_sampled


    def extract_superevent_features_for_dir(
        self,
        data_loader: torch.utils.data.DataLoader,
        out_dir: Path,
        model,
        config: dict,
        device: torch.device
    ):
        """
        Extract SuperEvent keypoints/descriptors from all MCTS frames in the dataset.
        Writes mcts_00000.feat.npz, mcts_00001.feat.npz, ... into out_dir.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = data_loader.dataset

        # Get topk and offsets from dataset
        top_k = dataset.get_topk()
        off_top, off_left, off_bottom, off_right = dataset.get_offsets()

        # Original image and cropped shapes from dataset
        H, W = dataset.H, dataset.W
        Hc, Wc = dataset.Hc, dataset.Wc

        # Global frame counter for naming: mcts_00000, mcts_00001, ...
        frame_idx = 0

        for batch in tqdm(data_loader, desc="SuperEvent keypoint extraction", unit="batch"):
            # batch: (B, C, Hc, Wc)
            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch)

                # Older versions might return a tuple
                if isinstance(pred, tuple):
                    pred = {"prob": pred[0], "descriptors": pred[1]}

                prob = pred["prob"]            # (B, 1, Hc, Wc)
                desc_map = pred["descriptors"] # (B, D, Hc, Wc)

                # NMS to get keypoints + scores per image in batch
                kpts_all, scores_all = fast_nms(prob, config, top_k=top_k)

            B = prob.shape[0]

            for b in range(B):
                # Build per-frame output path
                out_path = out_dir / f"mcts_{frame_idx:05d}.feat.npz"
                frame_idx += 1

                # Handle no-keypoint case
                if len(kpts_all[b]) == 0:
                    np.savez_compressed(
                        out_path,
                        keypoints=np.empty((0, 2), dtype=np.float32),
                        scores=np.empty((0,), dtype=np.float32),
                        descriptors=np.empty((0, desc_map.shape[1]), dtype=np.float32),
                        image_shape=np.array([H, W], dtype=np.int32),
                    )
                    continue

                # Keypoints and scores for this image
                kpts = kpts_all[b]      # (N, 2) (y, x) in cropped coords
                scores = scores_all[b]  # (N,)

                # Descriptors for this image: slice batch dimension
                desc_single = desc_map[b:b+1]  # (1, D, Hc, Wc)

                # Sample descriptors at keypoints
                desc_sampled = self.sample_descriptors_at_kpts(
                    kpts.float().to(device),
                    desc_single,
                    Hc,
                    Wc,
                )  # (N, D)

                # Undo crop: back to original full image coords
                kpts_np = kpts.float().cpu().numpy()
                kpts_np[:, 0] += off_top   # y
                kpts_np[:, 1] += off_left  # x

                xs = kpts_np[:, 1]
                ys = kpts_np[:, 0]
                kpts_xy = np.stack([xs, ys], axis=-1).astype(np.float32)  # (N, 2), (x, y)

                # Save per-frame NPZ, same structure as original
                np.savez_compressed(
                    out_path,
                    keypoints=kpts_xy,
                    scores=scores.cpu().numpy().astype(np.float32),
                    descriptors=desc_sampled.cpu().numpy().astype(np.float32),
                    image_shape=np.array([H, W], dtype=np.int32),
                )

    def extract_keypoints(self):
        """
        Run SuperEvent on the reference and query MCTS sequences
        and write mcts_*.feat.npz into self.reference_keypoints / self.query_keypoints.
        """
        device = self.device  # already set in __init__

        # Resolve SuperEvent config/weights.
        superevent_root = Path(THIS_DIR) / "external" / "superevent"
        config_path = superevent_root / "config" / "super_event.yaml"
        weights_path = superevent_root / "saved_models" / "super_event_weights.pth"

        if not config_path.exists():
            raise FileNotFoundError(f"SuperEvent config not found at {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"SuperEvent weights not found at {weights_path}")

        # Build model + config
        model, se_config = self.build_superevent_model(config_path, weights_path, device)

        # MCTS directories (where gen_mcts wrote frames)
        root = Path(self.data_root)
        ref_mcts_dir = root / self.dataset / self.reference / f"mcts_{self.reference}_{self.mcts_time[-1]}"
        query_mcts_dir = root / self.dataset / self.query / f"mcts_{self.query}_{self.mcts_time[-1]}"

        if not ref_mcts_dir.exists():
            raise FileNotFoundError(f"Reference MCTS directory not found: {ref_mcts_dir}")
        if not query_mcts_dir.exists():
            raise FileNotFoundError(f"Query MCTS directory not found: {query_mcts_dir}")

        # Output directories for keypoints (already used by rerank/keypoint_inference)
        self.reference_keypoints = os.path.join(self.keypoint_out, self.dataset, f"kps_{self.reference}")
        self.query_keypoints = os.path.join(self.keypoint_out, self.dataset, f"kps_{self.query}")

        ref_kp_dir = Path(self.reference_keypoints)
        query_kp_dir = Path(self.query_keypoints)

        # Define the dataloaders
        ref_dataset = EventGeMMCTS(str(ref_mcts_dir), se_config)
        query_dataset = EventGeMMCTS(str(query_mcts_dir), se_config)
        ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=self.keypoint_batch_size, shuffle=False, num_workers=4)
        query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=self.keypoint_batch_size, shuffle=False, num_workers=4)

        # Reference
        if not ref_kp_dir.exists():
            self.extract_superevent_features_for_dir(
                data_loader=ref_loader,
                out_dir=ref_kp_dir,
                model=model,
                config=se_config,
                device=device
            )

        # Query
        if not query_kp_dir.exists():
            self.extract_superevent_features_for_dir(
                data_loader=query_loader,
                out_dir=query_kp_dir,
                model=model,
                config=se_config,
                device=device
            )

    # ----------------------------------------------------------
    # Depth helpers (kept consistent)
    # ----------------------------------------------------------

    class _DepthLRUCache:
        """Tiny LRU cache for *downsampled* reference depth maps."""
        def __init__(self, max_items: int = 512):
            from collections import OrderedDict
            self.max_items = int(max_items)
            self._d = OrderedDict()

        def get(self, key: int):
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            return None

        def put(self, key: int, value: np.ndarray):
            self._d[key] = value
            self._d.move_to_end(key)
            if len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def _load_depth_map(self, depth_dir: Path, idx: int, pattern: str, index_offset: int = 0, down_hw=None):
        """
        Load a single depth PNG (expected 16-bit), return float32.
        Optionally downsamples to down_hw=(H,W) for speed.
        """
        import cv2
        path = depth_dir / pattern.format(idx + index_offset)
        if not path.exists():
            return None
        D = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if D is None:
            return None
        if D.ndim == 3:
            D = D[..., 0]
        D = D.astype(np.float32)
        if down_hw is not None:
            Ht, Wt = int(down_hw[0]), int(down_hw[1])
            D = cv2.resize(D, (Wt, Ht), interpolation=cv2.INTER_AREA)
        return D

    def _huber_mean(self, x: np.ndarray, delta: float) -> float:
        ax = np.abs(x)
        quad = np.minimum(ax, delta)
        lin = ax - quad
        return float(np.mean(0.5 * quad * quad + delta * lin))

    def _depth_affine_distance_resized(self, Rr: np.ndarray, Qr: np.ndarray) -> float:
        """
        Per-pair scale+shift align Q->R and compute robust residual.
        Assumes Rr and Qr are already resized to the same shape.
        """
        if Rr is None or Qr is None:
            return np.inf

        valid = np.isfinite(Rr) & np.isfinite(Qr) & (Rr > 0) & (Qr > 0)
        if valid.sum() < 500:
            return np.inf

        r = Rr[valid].reshape(-1)
        q = Qr[valid].reshape(-1)

        A = np.stack([q, np.ones_like(q)], axis=1)
        (s, t), *_ = np.linalg.lstsq(A, r, rcond=None)

        diff = (s * Qr + t - Rr)[valid]

        spread = float(np.percentile(r, 95) - np.percentile(r, 5))
        delta = max(1e-6, 0.05 * spread)
        return self._huber_mean(diff, delta)

    # ----------------------------------------------------------
    # Keypoint rerank (memory-safe)
    # ----------------------------------------------------------

# ----------------------------------------------------------
    # Keypoint rerank (Parallelized)
    # ----------------------------------------------------------

    def rerank(self, kp_pattern="mcts_{:05d}.feat.npz"):
        """
        Re-rank the top-k candidates using 2D-homology/inliers.
        Parallelized using joblib for speed.
        """
        R, Q = self.original.shape

        # Pre-load all query features into memory to avoid redundant disk I/O in workers
        queries_data = []
        for i in tqdm(range(Q), desc="Pre-loading query features"):
            queries_data.append(
                load_event_features(Path(self.query_keypoints), i, kp_pattern)
            )

        # Run parallel re-ranking
        results = Parallel(n_jobs=-1)(
            delayed(process_single_query)(
                q_idx=i,
                base_dists=self.original[:, i],
                top_k=self.top_k,
                q_data=queries_data[i],
                ref_kp_dir=Path(self.reference_keypoints),
                kp_pattern=kp_pattern,
                ransac_thresh=self.ransac_thresh,
                inlier_weight=self.inlier_weight,
            )
            for i in tqdm(range(Q), desc="Re-ranking (Keypoints)")
        )

        # Assemble results
        self.keypoint_reranked = self.original.copy()
        for q_idx, new_col in results:
            self.keypoint_reranked[:, q_idx] = new_col

    # ----------------------------------------------------------
    # Updated Depth Helpers
    # ----------------------------------------------------------

    def _compute_robust_depth_distance(self, Rr: np.ndarray, Qr: np.ndarray) -> float:
        """
        Computes a structural distance between two depth maps.
        Uses SSIM on normalized, masked depth to handle viewpoint shifts.
        """
        if Rr is None or Qr is None:
            return np.inf

        # 1. Create a mask of valid pixels (non-zero in both)
        mask = (Rr > 0) & (Qr > 0)
        
        # Safety check: if overlap is non-existent, it's a mismatch
        if np.sum(mask) < 100: 
            return 1.0

        # 2. Normalize intensity to [0, 1] based on valid percentiles
        def normalize_depth(D, m):
            vals = D[m]
            if len(vals) == 0: return D
            v_min, v_max = np.percentile(vals, [1, 99])
            # Avoid division by zero
            denom = v_max - v_min if v_max > v_min else 1.0
            return np.clip((D - v_min) / denom, 0, 1)

        Rr_n = normalize_depth(Rr, mask)
        Qr_n = normalize_depth(Qr, mask)

        # 3. Compute SSIM
        # mssim: the global average (scalar)
        # ssim_map: the per-pixel similarity map (array)
        mssim, ssim_map = ssim(Rr_n, Qr_n, full=True, data_range=1.0, win_size=7)
        
        # 4. Correctly index the MAP, not the scalar
        valid_scores = ssim_map[mask]
        
        if valid_scores.size == 0:
            return 1.0
            
        avg_structural_sim = np.mean(valid_scores)
        
        # Return 1.0 (worst) to 0.0 (perfect structural match)
        return 1.0 - float(avg_structural_sim)
    # ----------------------------------------------------------
    # Updated rerank_depth
    # ----------------------------------------------------------

    def rerank_depth(self):
        """
        Improved depth rerank using Structural Similarity and 
        absolute thresholding to prevent R@1 degradation.
        """
        R, Q = self.original.shape

        ref_depth_dir = Path(getattr(self, "reference_depth_dir"))
        qry_depth_dir = Path(getattr(self, "query_depth_dir"))

        depth_pattern = getattr(self, "depth_pattern", "depth_{:06d}.png")
        depth_index_offset = int(getattr(self, "depth_index_offset", 0))
        depth_weight = float(getattr(self, "depth_weight", 0.15))
        depth_down_hw = getattr(self, "depth_down_hw", (28, 28))
        depth_query_batch_size = int(getattr(self, "depth_query_batch_size", 64))
        depth_ref_cache_size = int(getattr(self, "depth_ref_cache_size", 512))

        outdir = os.path.join(self.feature_out, self.dataset, f"{self.reference}-{self.query}")
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, f"{self.dataset}_{self.reference}_{self.query}_rerank_depth_dist.npy")

        new_dist = np.memmap(out_path, dtype=np.float32, mode="w+", shape=(R, Q))
        K = int(self.top_k)
        cache = self._DepthLRUCache(max_items=depth_ref_cache_size)

        # if the re-rank is both
        if self.rerank_mode == "both":
            original = self.keypoint_reranked.copy()
        else:
            original = self.original.copy()

        for q0 in tqdm(range(0, Q, depth_query_batch_size), desc="Re-ranking (depth)", unit="batch"):
            q1 = min(Q, q0 + depth_query_batch_size)
            B = q1 - q0

            base_block = original[:, q0:q1].astype(np.float32, copy=False)
            new_dist[:, q0:q1] = base_block

            # Top-K indices
            part = np.argpartition(base_block, kth=K - 1, axis=0)[:K, :].astype(np.int32)
            topk_idx = np.empty_like(part)
            for bi in range(B):
                inds = part[:, bi]
                topk_idx[:, bi] = inds[np.argsort(base_block[inds, bi])]

            # Load query depths
            q_depths = [self._load_depth_map(qry_depth_dir, qi, depth_pattern, depth_index_offset, depth_down_hw) 
                        for qi in range(q0, q1)]

            for bi, qi in enumerate(range(q0, q1)):
                qD = q_depths[bi]
                if qD is None: continue

                refs = topk_idx[:, bi]
                errs = np.zeros(len(refs), dtype=np.float32)

                for j, r_idx in enumerate(refs):
                    r_idx_int = int(r_idx)
                    rD = cache.get(r_idx_int)
                    if rD is None:
                        rD = self._load_depth_map(ref_depth_dir, r_idx_int, depth_pattern, depth_index_offset, depth_down_hw)
                        if rD is not None: cache.put(r_idx_int, rD)
                    
                    # Use the new robust distance
                    errs[j] = self._compute_robust_depth_distance(rD, qD)

                # --- IMPROVED SCORING LOGIC ---
                # Using a fixed tau (0.2 - 0.5) ensures we only boost matches that are 
                # actually structurally similar, rather than just 'the best of a bad bunch'.
                tau = 0.3 
                sims = np.exp(-errs / tau).astype(np.float32)

                # Optional: Only apply boost if SSIM distance is reasonably low (e.g. < 0.7)
                mask_good = errs < 0.7
                new_dist[refs[mask_good], qi] = base_block[refs[mask_good], bi] - (depth_weight * sims[mask_good])

        new_dist.flush()
        self.depth_reranked = new_dist

    def keypoint_inference(self):
        """
        Orchestrates the re-ranking pipeline based on self.rerank_mode.
        """
        # 1. Load Global Similarity Matrix
        sim_matrix_path = os.path.join(
            self.feature_out, 
            self.dataset, 
            f"{self.reference}-{self.query}", 
            f"{self.dataset}_{self.reference}_{self.query}_similarity.pt"
        )
        
        if not os.path.exists(sim_matrix_path):
            raise FileNotFoundError(f"Similarity matrix not found: {sim_matrix_path}")

        sim = torch.load(sim_matrix_path, map_location="cpu")
        self.original = (1.0 - sim.float()).numpy()
        del sim # Free memory

        mode = getattr(self, "rerank_mode", "keypoints") # "keypoints" | "depth" | "both"

        # 2. Handle Depth-Only Mode
        if mode == "depth":
            self.rerank_depth()
            return self.original, None, self.depth_reranked

        # 3. Keypoint/Both Mode: Ensure MCTS files exist
        root = self.data_root
        self.reference_path = os.path.join(root, self.dataset, self.reference, f"mcts_{self.reference}_{self.mcts_time[-1]}")
        self.query_path = os.path.join(root, self.dataset, self.query, f"mcts_{self.query}_{self.mcts_time[-1]}")

        # Ensure keypoints are extracted
        if (not os.path.exists(self.reference_path)) or (not os.path.exists(self.query_path)):
            gen_mcts(root, self.dataset, self.reference, self.query, self.mcts_time)
        
        # Ensure keypoint directories are defined
        self.reference_keypoints = os.path.join(self.keypoint_out, self.dataset, f"kps_{self.reference}")
        self.query_keypoints = os.path.join(self.keypoint_out, self.dataset, f"kps_{self.query}")

        if (not os.path.exists(self.reference_keypoints)) or (not os.path.exists(self.query_keypoints)):
            self.extract_keypoints()

        # 4. Perform Keypoint Re-ranking
        self.rerank()

        # 5. Optional: Chain Depth Re-ranking
        if mode == "both":
            # Use keypoint-refined distances as the base for depth refinement
            self.distance_matrix = self.keypoint_reranked.copy()
            self.rerank_depth()

            return self.original, self.keypoint_reranked, self.depth_reranked
        
        return self.original, self.keypoint_reranked, None