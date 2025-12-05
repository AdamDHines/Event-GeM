'''
Imports
'''
import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
from eventgem.dataset import EventGeMData
from eventgem.utils.eventlab_config import update_config, eventlab_data
from eventgem.utils.rerank_utils import load_event_features, process_single_query

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

    def GeM(self, feats, p=3.0):
        return F.avg_pool2d((feats.clamp(min=1e-6)).pow(p), (feats.shape[-2], feats.shape[-1])).pow(1.0/p)

    def extract_features(self):
        # Define backbone model
        backbone = vit_contrastive_patch16_small(mask_ratio=0.0, in_chans=2, num_classes=512)
        checkpoint = torch.load(self.backbone_ckpt, map_location='cpu')

        # Load checkpoint
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

        # Remove prefixes
        new_state_dict = {k.replace("encoder_q.", "").replace("module.", ""): v for k, v in state_dict.items()}
        
        backbone.load_state_dict(new_state_dict, strict=False)
        backbone.to(self.device).eval()

        # Define the DataLoaders
        ref_dataset = EventGeMData(self.reference_path)
        query_dataset = EventGeMData(self.query_path)
        ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

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
        self.reference_path = os.path.join(root, self.reference, f"{self.reference}-frames-{self.recon_msec}")
        self.query_path = os.path.join(root, self.query, f"{self.query}-frames-{self.recon_msec}")

        if not os.path.exists(self.reference_path):
            # Updat the eventlab config and generate the data
            update_config(root, self.dataset, self.reference, self.query, time=self.recon_msec)
            eventlab_data() # generates the data with specified arguments
        if not os.path.exists(self.query_path):
            raise FileNotFoundError(f"Query directory '{self.query_path}' does not exist - something went wrong with the Event-LAB Data generation.")
        
        # Check if features have been pre-computed
        outdir = os.path.join(self.feature_out, self.dataset, f"{self.reference}-{self.query}")
        feat_ref_path = os.path.join(outdir, f"{self.dataset}_{self.reference}_features.pt")
        feat_query_path = os.path.join(outdir, f"{self.dataset}_{self.query}_features.pt")
        if not os.path.exists(feat_ref_path) and not os.path.exists(feat_query_path):
            # Perform feature extraction
            self.extract_features()

    def extract_keypoints(self):
        pass

    def rerank(self, kp_pattern="mcts_{:04d}.feat.npz"):
        R, Q = self.distance_matrix.shape

        # Pre-load queries exactly like before
        queries_data = []
        for i in tqdm(range(Q)):
            queries_data.append(
                load_event_features(Path(self.query_keypoints), i, kp_pattern)
            )

        results = Parallel(n_jobs=-1)(
            delayed(process_single_query)(
                q_idx=i,
                base_dists=self.distance_matrix[:, i],
                top_k=self.top_k,
                q_data=queries_data[i],
                ref_kp_dir=Path(self.reference_keypoints),
                kp_pattern=kp_pattern,
                ransac_thresh=self.ransac_thresh,
                inlier_weight=self.inlier_weight,
            )
            for i in tqdm(range(Q), desc="Re-ranking")
        )

        self.new_dist_matrix = self.distance_matrix.copy()
        for q_idx, new_col in results:
            self.new_dist_matrix[:, q_idx] = new_col


    def keypoint_inference(self):
        # Check if the similarity matrix for specified dataset exists
        sim_matrix_path = os.path.join(self.feature_out, self.dataset, f"{self.reference}-{self.query}", f"{self.dataset}_{self.reference}_{self.query}_similarity.pt")
        if not os.path.exists(sim_matrix_path):
            raise FileNotFoundError(f"Similarity matrix '{sim_matrix_path}' not found. Please run feature extraction first.")

        # Load the similarity matrix
        self.sim_matrix = torch.load(sim_matrix_path)
        # Convert to distance matrix
        self.distance_matrix = (1.0 - self.sim_matrix).numpy()

        # Check that specified datasets exist - need MCTS reconstructued directories
        root = self.data_root
        self.reference_path = os.path.join(root, self.reference, f"mcts_{self.reference}")
        self.query_path = os.path.join(root, self.query, f"mcts_{self.query}")

        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(f"Reference directory '{self.reference_path}' does not exist.")
        if not os.path.exists(self.query_path):
            raise FileNotFoundError(f"Query directory '{self.query_path}' does not exist.")
        
        # Check if keypoints have been pre-computed
        self.reference_keypoints = os.path.join(self.keypoint_out, self.dataset, f"kps_{self.reference}")
        self.query_keypoints = os.path.join(self.keypoint_out, self.dataset, f"kps_{self.query}")

        if not os.path.exists(self.reference_keypoints) or not os.path.exists(self.query_keypoints):
            # Perform keypoint extraction
            self.extract_keypoints()

        # Re-rank the top-k candidates using 2D-homology
        self.rerank()

        return self.distance_matrix, self.new_dist_matrix
