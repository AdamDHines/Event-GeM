'''
Imports
'''
import os
import sys
import torch
import yaml
import gc

import cv2
import numpy as np
import eventcv as ecv
import torch.nn.functional as F

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # optional; without it BLAS keeps its own thread count and rerank is slower
    from contextlib import contextmanager

    @contextmanager
    def threadpool_limits(limits=None, user_api=None):
        yield

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from huggingface_hub import hf_hub_download
from eventgem.inference import stream_file
from eventgem.training.common import ResidualHead
from eventgem.dataset import EventGeMMCTS
from eventgem.utils.eventlab_config import update_config
from eventgem.utils.rerank_utils import process_single_query, open_keypoint_bank, bank_lookup
from eventgem.utils.kp_store import KeypointStoreWriter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Superevent root — where the "models" package lives
SUPEREVENT_ROOT = os.path.join(THIS_DIR, "external", "superevent")

for path in (SUPEREVENT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

# Now "models" resolves to eventgem/external/superevent/models
from models.super_event import SuperEvent, SuperEventFullRes
from models.util import fast_nms

import matplotlib.pyplot as plt


def pool_grid_gamma(feats, gamma, g=4):
    """
    Regional g x g GeM with a per-cell exponent. [B,128,H,W] -> [B, g*g*128], L2-normalised.

    The shipped head was trained on this descriptor, so three details here are load-bearing and
    are not tidy-up candidates:

      * `astype(int)` truncates. On the 30x40 FPN grid that gives row bounds [0,7,15,22,30]
        (heights 7, 8, 7, 8); rounding or `//` would move two of the four rows.
      * the per-window mean is clamped before `pow(1/p)`. At the learned gamma ~19 a near-empty
        window underflows `mean(xn^p)` into the subnormal range.
      * the max-rescale is GLOBAL per (batch, channel), not per cell, so every block keeps a
        common scale reference and encodes *where* rather than *what*.

    Cells are emitted row-major, so gamma[i*g + j] pairs with descriptor columns 128k : 128(k+1).
    """
    x = feats.clamp(min=1e-6)
    m = x.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    xn = x / m
    ys = np.linspace(0, x.shape[-2], g + 1).astype(int)
    xs = np.linspace(0, x.shape[-1], g + 1).astype(int)
    parts = []
    for i in range(g):
        for j in range(g):
            pij = gamma[i * g + j]
            q = xn[..., ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            mp = q.pow(pij).mean(dim=(-2, -1)).clamp(min=1e-12)
            parts.append(mp.pow(1.0 / pij) * m[..., 0, 0])
    out = torch.stack(parts, dim=1)                                   # [B, g*g, 128]
    out = F.normalize(out, p=2, dim=2).reshape(out.shape[0], -1)      # per-block L2
    return F.normalize(out, p=2, dim=1)                               # global L2


class EventGeM:
    def __init__(self, args):
        # Before running, ensure repository was cloned with --recurse-submodules
        submodule_paths = ["./eventgem/external/superevent", "./eventgem/external/eventlab"]
        for path in submodule_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Submodule path '{path}' not found. Please clone Event-GeM with --recurse-submodules.")

        # Set all args as class attributes
        for k in vars(args): 
            setattr(self, k, getattr(args, k))

        # Set args as self.args for easy access in streaming mode
        self.args = args

        # Get and set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_head(self, device):
        """
        The shipped projection head, downloaded on first use.

        `eventgem.pt` is a 4x4 regional descriptor head: a dim-preserving residual MLP over a
        2048-D (16 x 128) pooled vector, plus `grid`, the 16 learned GeM exponents it was trained
        with (tied per grid row -- sky 6.69, skyline 17.12, facade 17.61, road 19.19). It sits on
        top of the SuperEvent trunk from the submodule rather than replacing it.

        `hf_hub_download` caches, resumes and honours HF_HUB_OFFLINE, and `load_state_dict` is
        strict, so a wrong checkpoint fails here rather than silently mis-pooling.
        """
        ckpt = torch.load(hf_hub_download("AdamHines/eventgem", "eventgem.pt"),
                          map_location=device, weights_only=False)
        head = ResidualHead(2048, 1638).to(device)
        head.load_state_dict(ckpt["head"])
        gamma = torch.as_tensor(ckpt["grid"], dtype=torch.float32, device=device).ravel()
        print(f"[INFO] EventGeM head loaded (gamma rows {[round(float(v), 2) for v in gamma[::4]]})")
        return head.eval(), gamma

    def extract_superevent_features(self, sim_file):
        """
        Global descriptors from SuperEvent's pre-head feature map, as a drop-in
        replacement for the ECDPT ViT.

        `SuperEvent.forward` computes `features = self.fpn(self.backbone(x))` and only then
        branches to the detector and descriptor heads. GeM pooling that shared map yields a
        128-D global descriptor from the same forward pass keypoint extraction already runs,
        so this path removes the ECDPT backbone entirely rather than adding to it.

        Descriptors come from the MCTS frames (the input SuperEvent expects), not the
        polarity frames the ECDPT path consumes, so the resulting matrix can differ by a
        frame or two per sequence.
        """
        device = self.device

        model, se_config = self.build_superevent_model(
            Path(self.se_config), Path(self.se_weights), device
        )
        head, gamma = self.load_head(device)
        ref_dir = f"{self.data_root}/{self.dataset}/{self.reference}/{self.reference}.hdf5"
        query_dir = f"{self.data_root}/{self.dataset}/{self.query}/{self.query}.hdf5"

        ref_dataset = EventGeMMCTS(
            ref_dir, se_config, offset=self.ref_offset,
            dt_ms=self.dt_ms, max_window_ms=getattr(self, "max_window_ms", None),
        )
        query_dataset = EventGeMMCTS(
            query_dir, se_config, offset=self.query_offset,
            dt_ms=self.dt_ms, max_window_ms=getattr(self, "max_window_ms", None),
        )
        self.ref_loader = torch.utils.data.DataLoader(
            ref_dataset, batch_size=self.keypoint_batch_size, shuffle=False, num_workers=4, collate_fn=ecv.collate
        )
        self.query_loader = torch.utils.data.DataLoader(
            query_dataset, batch_size=self.keypoint_batch_size, shuffle=False, num_workers=4, collate_fn=ecv.collate
        )

        top_k = ref_dataset.get_topk()
        off_top, off_left, _, _ = ref_dataset.get_offsets()
        H, W = ref_dataset.H, ref_dataset.W
        Hc, Wc = ref_dataset.Hc, ref_dataset.Wc

        def pool_loader(loader, se_config, top_k, off_top, off_left, H, W, Hc, Wc, seq, desc):
            """
            One pass over a sequence: GeM global descriptors *and* keypoints, from a single
            forward of the shared trunk.

            Global descriptors are small enough to accumulate (N x 128 floats), so they are
            returned. Keypoints are not -- at up to top_k per frame over >14k frames the padded
            descriptor block runs to gigabytes -- so they are streamed straight into one
            memory-mapped `<seq>_kps.pt` store under `self.kps_out`, a batch at a time. Nothing
            per-frame is retained here and nothing lands on the GPU past the batch it came from.
            """
            n_frames = len(loader.dataset)
            desc_dim = int(se_config["descriptor_size"])
            writer = KeypointStoreWriter(
                out_path=os.path.join(self.kps_out, f"{self.dataset}_{seq}_kps.pt"),
                n_frames=n_frames,
                k_max=top_k,
                desc_dim=desc_dim,
                image_shape=(H, W),
            )

            feats = []
            try:
                with torch.inference_mode():
                    for batch in tqdm(loader, desc=desc, unit="batch"):
                        batch = batch.to(device, non_blocking=True)
                        # The trunk is the dominant cost (4.0 ms/frame vs 0.1 for the detector),
                        # so it is run once and both heads branch off it, exactly as
                        # SuperEvent.forward does -- calling model(batch) here would pay for the
                        # backbone and FPN a second time.
                        features = model.fpn(model.backbone(batch))
                        pooled = pool_grid_gamma(features.float(), gamma)
                        feats.append(pooled.detach().cpu())

                        _, prob = model.detector(features)      # (B, Hc, Wc)
                        _, desc_map = model.descriptor(features) # (B, D, Hc, Wc)

                        # NMS to get keypoints + scores per image in batch
                        kpts_all, scores_all = fast_nms(prob, se_config, top_k=top_k)

                        B = batch.shape[0]

                        # One padded block per batch, filled on the GPU and copied down in a
                        # single transfer rather than three small copies per frame.
                        blk_kpts = torch.zeros(B, top_k, 2, device=device)
                        blk_desc = torch.zeros(B, top_k, desc_dim, device=device)
                        blk_scores = torch.zeros(B, top_k, device=device)
                        blk_counts = torch.zeros(B, dtype=torch.int32)

                        for b in range(B):
                            n = min(len(kpts_all[b]), top_k)
                            if n == 0:
                                continue

                            # Keypoints and scores for this image
                            kpts = kpts_all[b][:n].float()  # (n, 2) (y, x) in cropped coords
                            scores = scores_all[b][:n]      # (n,)

                            # Sample descriptors at keypoints, from this image's slice of the map
                            blk_desc[b, :n] = self.sample_descriptors_at_kpts(
                                kpts.to(device),
                                desc_map[b:b + 1],
                                Hc,
                                Wc,
                            )  # (n, D)

                            # Undo crop: back to original full image coords, as (x, y)
                            blk_kpts[b, :n, 0] = kpts[:, 1] + off_left  # x
                            blk_kpts[b, :n, 1] = kpts[:, 0] + off_top   # y
                            blk_scores[b, :n] = scores
                            blk_counts[b] = n

                        # .cpu() so the store is device-agnostic downstream
                        writer.add_batch(
                            keypoints=blk_kpts.cpu(),
                            descriptors=blk_desc.float().cpu(),
                            scores=blk_scores.float().cpu(),
                            counts=blk_counts,
                        )

                writer.save()
            finally:
                writer.cleanup()

            return torch.cat(feats, dim=0)

        ref_feats = pool_loader(self.ref_loader, se_config, top_k, off_top, off_left, H, W, Hc, Wc,
                                self.reference, "Extracting reference features (SuperEvent)")
        torch.save(ref_feats, os.path.join(self.outdir, f"{self.dataset}_{self.reference}_features.pt"))

        query_feats = pool_loader(self.query_loader, se_config, top_k, off_top, off_left, H, W, Hc, Wc,
                                  self.query, "Extracting query features (SuperEvent)")
        torch.save(query_feats, os.path.join(self.outdir, f"{self.dataset}_{self.query}_features.pt"))

        print(f"[INFO] SE-GeM descriptors: ref {tuple(ref_feats.shape)} | query {tuple(query_feats.shape)}")

        # The head is fit-free -- it was trained on the pooled descriptor's own statistics, with
        # no PCA and no whitening -- and its forward already returns an L2-normalised vector.
        with torch.inference_mode():
            ref_feats = head(ref_feats.to(device)).cpu()
            query_feats = head(query_feats.to(device)).cpu()

        # Same cosine convention as the ECDPT path: rows = references, cols = queries.
        sim_matrix = torch.matmul(query_feats, ref_feats.t()).T
        torch.save(sim_matrix, sim_file)

        # delete the matrices for memory
        del ref_feats
        del query_feats
        del sim_matrix

        # Drop the loaders too: they hold the open event streams and their worker state, and
        # nothing downstream of extraction needs them. Reranking starts immediately after this
        # and is the peak-memory phase of the run.
        self.ref_loader = self.query_loader = None
        del model

        # clear cuda cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def feature_inference(self):
        # Check that the specified datasets exist - need frame reconstructued directories
        root = self.data_root
        self.reference_path = os.path.join(root, self.dataset, self.reference, f"{self.reference}-frames-{self.dt_ms}")
        self.query_path = os.path.join(root, self.dataset, self.query, f"{self.query}-frames-{self.dt_ms}")
        # update_config(root, self.dataset, self.reference, self.query, time=self.dt_ms, stream=self.stream, demo=self.demo)
        # Run feature extraction for reference and query sets. The SuperEvent global
        # descriptors live in their own directory so they never clobber the ECDPT baseline.
        pair_dir = f"{self.reference}-{self.query}"
        self.outdir = os.path.join(self.feature_out, self.dataset, pair_dir)
        self.kps_out = os.path.join(self.keypoint_out, self.dataset, pair_dir)
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.kps_out, exist_ok=True)
        sim_file = os.path.join(self.outdir, f"{self.dataset}_{self.reference}_{self.query}_similarity.pt")
        self.sim_file = sim_file
        if self.rerun_features or not os.path.exists(sim_file) and not self.stream:
            self.extract_superevent_features(sim_file)
        elif self.stream:
            print("[INFO] Running in streaming mode. Extracting features on-the-fly without saving to disk.")
            stream_file(self.args)  # This will run the streaming inference logic defined in inference.py
        else:
            print("[INFO] Skipping feature extraction (already exists). Set --rerun-features to force re-extraction.")

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

               # self.debug_plot_keypoints(batch[b], kpts_xy, save_path=f"debug_kpts_{frame_idx-1:05d}.png")

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
    # Keypoint rerank (Parallelized)
    # ----------------------------------------------------------

    def kp_store_path(self, seq):
        """Path of the single-file keypoint store for one sequence, as written by `pool_loader`."""
        return os.path.join(self.kps_out, f"{self.dataset}_{seq}_kps.pt")

    def rerank(self):
        """
        Re-rank the top-k candidates using 2D-homology/inliers.
        Parallelized using joblib for speed.

        Both sides come from the `<seq>_kps.pt` stores written during feature extraction. They are
        already packed and padded, so there is no per-frame file to open and no bank to build:
        the reference store is memory-mapped once per worker (see `open_keypoint_bank`) and query
        frames are cut out of the query store as tasks are dispatched, which keeps resident memory
        at a few frames rather than the whole ~2 GB traverse.
        """
        R, Q = self.original.shape

        ref_store = self.kp_store_path(self.reference)
        query_store = self.kp_store_path(self.query)
        for path in (ref_store, query_store):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Keypoint store not found: {path}. Re-run feature extraction "
                    f"(--rerun-features) to write it.")

        query_bank = open_keypoint_bank(query_store)
        n_query_frames = len(query_bank[2])
        if n_query_frames != Q:
            print(f"[WARN] query store has {n_query_frames} frames but the similarity matrix has "
                  f"{Q} columns; reranking the overlap only")

        # Assemble into the output up front so completed columns can be written straight into it.
        self.keypoint_reranked = self.original.copy()

        # Threads, not processes. The matching inner loop is BLAS and cv2, both of which release
        # the GIL, so a thread pool parallelises it just as well -- and it does so in one address
        # space. joblib's default loky backend would instead start eight interpreters that each
        # import torch and each mmap the ~2 GB reference store: measured at +3.9 GB peak against
        # +18 MB for threads, and that spike lands immediately after extraction has filled the
        # page cache, which is what was getting the run killed by systemd-oomd.
        #
        # BLAS is pinned to one thread per worker: the per-pair gemm is only 160x256x160, where
        # MKL's own threading buys nothing and eight workers x eight MKL threads collapses into
        # oversubscription (measured 9.10 -> 5.29 ms/query when pinned).
        #
        # `return_as` streams results back rather than collecting a list, which would hold a full
        # R-vector per query -- another copy of the whole distance matrix (~740 MB here).
        cv2.setNumThreads(1)
        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=-1, backend="threading", return_as="generator_unordered")(
                delayed(process_single_query)(
                    q_idx=i,
                    base_dists=self.original[:, i],
                    top_k=self.top_k,
                    q_data=bank_lookup(query_bank, i) if i < n_query_frames else None,
                    ref_store=ref_store,
                    ransac_thresh=self.ransac_thresh,
                    inlier_weight=self.inlier_weight,
                    match_filter=getattr(self, "match_filter", "ratio"),
                    match_ratio=float(getattr(self, "match_ratio", 0.8)),
                )
                for i in tqdm(range(Q), desc="Re-ranking (Keypoints)")
            )

            # Assemble results as they complete; each column is freed once written.
            for q_idx, new_col in results:
                self.keypoint_reranked[:, q_idx] = new_col

    def rerank_inference(self):
        # 1. Load Global Similarity Matrix (set by feature_inference, which knows which
        #    global backbone produced it)
        sim_matrix_path = getattr(self, "sim_file", None) or os.path.join(
            self.feature_out,
            self.dataset,
            f"{self.reference}-{self.query}",
            f"{self.dataset}_{self.reference}_{self.query}_similarity.pt"
        )

        if not os.path.exists(sim_matrix_path):
            raise FileNotFoundError(f"Similarity matrix not found: {sim_matrix_path}")

        # Distances in place: `1.0 - sim.float()` would hold the loaded matrix and its complement
        # at once, which is ~1.5 GB on brisbane_event for no reason.
        sim = torch.load(sim_matrix_path, map_location="cpu").float()
        sim.neg_().add_(1.0)
        self.original = sim.numpy()
        del sim # the numpy view keeps the storage alive

        # 4. Perform Keypoint Re-ranking
        self.rerank()
        
        return self.original, self.keypoint_reranked