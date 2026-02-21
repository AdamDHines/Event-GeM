#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import streamutils.stream as stream
import os

import sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from external.depthanyevent.models import fetch_model  # from the DepthAnyEvent repo
import imageio

THIS_DIR = Path(__file__).resolve().parent
BACKBONE_ROOT = THIS_DIR / "external" / "backbone"
SUPEREVENT_ROOT = THIS_DIR / "external" / "superevent"

# Order matters: backbone's `utils` must win
sys.path.insert(0, str(SUPEREVENT_ROOT))
sys.path.insert(0, str(BACKBONE_ROOT))

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # Dataset parameters
    ap.add_argument("--hdf5",  type=str, required=True, 
                    help="Path to the input hdf5 file containing events")
    ap.add_argument("--dataset",  type=str, required=True,
                    help="Name of the dataset")
    ap.add_argument("--reference", type=str, required=True,
                    help="Name of the reference")
    ap.add_argument("--query", type=str, required=True,
                    help="Name of the query")
    ap.add_argument("--time-scale", type=float, default=1e-9,
                    help="Time scale for event accumulation")
    ap.add_argument("--chunk-size", type=int, default=250_000,
                    help="Number of events to process in each chunk (for streaming from hdf5)")
    ap.add_argument("--start-time", type=float, default=None,
                    help="Start time for event processing (in seconds)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Maximum number of frames to process (None for no limit)")
    ap.add_argument("--infer-full-scan", action="store_true",
                help="Whether to run inference on the full scan of the dataset (instead of streaming in chunks)")

    # Method parameters
    ap.add_argument("--method", type=str,  default="eventgem", 
                choices=["eventgem", "eventgem-d", "ecdpt", "superevent", "lens", "sparse", "eventvlad"],
                help="Which method to run (for ablation or comparison)")
    ap.add_argument('--extract-reference', action='store_true', 
                    help="Whether to extract reference information (for initial mapping)")
    ap.add_argument('--features-dir', type=str, default='eventgem/features',
                    help="Directory to save extracted features (if --extract-reference is set)")
    ap.add_argument('--keypoint-dir', type=str, default='eventgem/keypoints',
                    help="Directory to save extracted keypoints (if --extract-reference is set)")
    ap.add_argument('--depth-dir', type=str, default='eventgem/depth',
                    help="Directory to save extracted depth maps (if --extract-reference is set)")
    
    # ViT backbone parameters
    ap.add_argument("--backbone-ckpt", type=str, default="./eventgem/ckpt/pr.pt",
                    help="Path to the ViT backbone checkpoint to load")
    ap.add_argument("--resize", type=str, default="nearest", choices=["nearest", "bilinear"],
                    help="Resize method to use when resizing input event frames for ViT (if needed)")
    ap.add_argument("--amp", action="store_true",
                    help="Whether to use automatic mixed precision for ViT inference (GPU only)")
    ap.add_argument("--dt-ms", type=float, default=50,
                    help="Time window (in ms) to accumulate events for each inference step")

    # SuperEvent parameters
    ap.add_argument("--retrieval-k", type=int, default=10,
                    help="Number of top candidates to retrieve from ViT before re-ranking")
    ap.add_argument("--se-config", type=str, default="eventgem/external/superevent/config/super_event.yaml",
                    help="Path to the SuperEvent config file")
    ap.add_argument("--se-weights", type=str, default="eventgem/external/superevent/saved_models/super_event_weights.pth",
                    help="Path to the SuperEvent weights file")
    ap.add_argument("--se-topk", type=int, default=170,
                    help="Number of top candidates to keep after re-ranking")
    ap.add_argument("--mcts-windows-ms", type=float, nargs="+", default=[10, 20, 30, 40, 50],
                    help="List of time windows (in ms) for MCTS")
    ap.add_argument("--ref-kp-pattern", type=str, default="mcts_{:05d}.feat.npz",
                    help="Pattern to match reference keypoint files (should include a placeholder for candidate ID)")
    ap.add_argument("--ref-kp-cache", type=int, default=2048,
                    help="Size of the cache for reference keypoints")
    ap.add_argument("--ransac-thresh", type=float, default=5.0,
                    help="RANSAC threshold for keypoint matching")
    ap.add_argument("--inlier-weight", type=float, default=0.05,
                    help="Weight for inliers in keypoint matching")
    ap.add_argument("--match-ratio", type=float, default=0.8,
                    help="Match ratio for keypoint matching")
    
    # Depth parameters
    ap.add_argument('--depth-loadmodel', default='eventgem/external/depthanyevent/models/rec_dav2/synth/synth.pth',
                        help='Path to model checkpoint')
    ap.add_argument('--depth-config', type=str, default='eventgem/external/depthanyevent/configs/test/recdav2/rec_dav2_mvsec_test.json',
                        help='Path to config file. If not specified, config from model folder/checkpoint is used')
    ap.add_argument('--depthp-resize', type=int, default=28,
                    help="Resize parameter for depth prediction (input will be resized to (depthp_resize, depthp_resize) before running depth model)")
    ap.add_argument("--depth-pattern", type=str, default="depth_{:06d}.png",
                    help="File pattern for depth maps")
    # Evaluation metrics and parameters
    ap.add_argument("--target-hz", type=float, default=20.0)
    ap.add_argument("--gt-file", type=str, default=None,
                    help="Path to the ground truth file")

    # Direct streaming parameters
    ap.add_argument('--live-davis', action='store_true',
                    help="Whether to run on live DAVIS stream instead of hdf5")

    args = ap.parse_args()

    # File path for hdf5
    if not args.live_davis:
        hdf5_path = Path(args.hdf5)
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)
    
    # If extracting reference information, set storage paths
    # if args.extract_reference:
    ref_feats_dir = Path(args.features_dir) / args.dataset / f"{args.reference}-{args.dt_ms}"
    ref_feats_file = ref_feats_dir / f"{args.dataset}_{args.reference}_features.pt"
    ref_kp_dir = Path(args.keypoint_dir) / args.dataset / f"{args.reference}-{args.dt_ms}"
    ref_depth_dir = Path(args.depth_dir) / args.dataset / f"{args.reference}-{args.dt_ms}"
    ref_feats_dir.mkdir(parents=True, exist_ok=True)
    ref_kp_dir.mkdir(parents=True, exist_ok=True)
    ref_depth_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracted reference features will be saved to: {ref_feats_dir}")
    print(f"[INFO] Extracted reference keypoints will be saved to: {ref_kp_dir}")
    print(f"[INFO] Extracted reference depth maps will be saved to: {ref_depth_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if not args.live_davis:
        with h5py.File(hdf5_path, "r") as f:
            x_dset, y_dset, t_dset, p_dset = stream.find_event_datasets(f)
            H, W = stream.infer_resolution(x_dset, y_dset, args.chunk_size, args.infer_full_scan)
            
            t0_raw = int(t_dset[0]); tN_raw = int(t_dset[-1])
            print(f"[INFO] Sensor: {H}x{W}")
    else:
        H, W = 260, 346  # default DAVIS resolution
        stream_davis = torch.cuda.Stream()

    print(args.method)
    # Load the corresponding model to run the inference on
    if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
        vit = stream.load_vit_backbone(args.backbone_ckpt, device)
        vitH, vitW = stream.infer_vit_input_hw(vit)
        print(f"[INFO] ViT expects ~ {vitH}x{vitW}")
        ref_db = stream.load_ref_vit_db(ref_feats_file, device=device, dtype=torch.float16 if args.amp else torch.float32)
        ref_db = ref_db.to(dtype=torch.float16)
        stream_vit = torch.cuda.Stream()

    if args.method in ["superevent", "eventgem", "eventgem-d"]:
        se_model, se_cfg = stream.build_superevent_model(Path(args.se_config), Path(args.se_weights), device)
        from models.util import fast_nms
        off_top, off_left, _, _, h_end, w_end, Hc, Wc = stream.compute_superevent_crop_offsets(H, W, se_cfg)
        print(f"[INFO] SuperEvent crop: top={off_top} left={off_left} -> {Hc}x{Wc}")
        stream_se = torch.cuda.Stream()
        windows_sec = torch.tensor(np.array(args.mcts_windows_ms, dtype=np.float32) * 1e-3, device=device)
        if args.method == "superevent":
            # Loading the reference keypoint descriptors
            ref_descs = stream.preload_ref_descs(ref_kp_dir, args.ref_kp_pattern)
            

    if args.method == "eventgem-d":
        # Load model + config
        depth_ckpt, depth_config = stream.load_and_merge_config(args.depth_loadmodel, args.depth_config)
        depth_model = fetch_model(depth_config['model'], args, device, test=True, _state_dict=depth_ckpt)
        model_name = depth_config['model']['model_type']
        depth_model.eval()
        stream_d = torch.cuda.Stream()
        prev_states=None

    if args.method == "lens":
        # model name
        model_name = '/home/adam/repo/Event-GeM/sunset2-frames-50_LENS_IN784_FN1568_DB12824.pth'
        lens_model = stream.LENS()
        lens_model.load_state_dict(torch.load(model_name), strict=False)
        stream_lens = torch.cuda.Stream()

    if args.method == "sparse":
        stream_sparse = torch.cuda.Stream()
        ref_dir = '/media/adam/vprdatasets/eventgem/brisbane_event/sunset2/sunset2-frames-50'
        # preload frames
        ref_npy = sorted(list(Path(ref_dir).glob("*.npy")))
        frames = np.array([np.load(p) for p in ref_npy])

        reference_data = [arr.sum(axis=2) for arr in frames]
        reference_data = np.array(reference_data)
        del frames
        reference_data_noburst = stream.remove_random_bursts(reference_data, threshold=10)
        reference_event_means = reference_data_noburst.mean(axis=0)

        prob_to_draw_from = stream.adjust_and_normalize_probabilities(reference_event_means)
        random_pixels = np.array(stream.get_random_pixels(100, 
                                            im_width=346, 
                                            im_height=260, 
                                            local_suppression_radius=7, 
                                            prob_to_draw_from=prob_to_draw_from))
        x_coords = random_pixels[:, 1]
        y_coords = random_pixels[:, 0]
        sparse_reference_data = reference_data[:, y_coords, x_coords]
        print(sparse_reference_data.shape)
        del reference_data
    
    if args.method == "eventvlad":
        stream_vlad = torch.cuda.Stream()
        # preload the reference descriptors
        ref_desc = np.load('/media/adam/vprdatasets/eventgem/brisbane_event/sunset2/sunset2_eventvlad.npy')
        denoise_model = stream.build_model('event_denoiser', dep_u=5, dep_s=5, slope=0.2).to(device)
        ckpt_path = '/home/adam/repo/Event-GeM/eventgem/external/eventlab/baselines/EventVLAD/denoiser_brisbane'
        denoise_model = stream.load_checkpoint_into_model(denoise_model, ckpt_path)
        query = None
        netvlad_model = stream.build_eventvlad_model_from_tar(
            weights_path="/home/adam/repo/Event-LAB/baselines/EventVLAD/vgg16_eventvlad.tar",
            num_clusters=64,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    ref_store = None
    if args.method in ["eventgem", "eventgem-d", "superevent"]:
        if ref_kp_dir is None:
            raise ValueError("--do-rerank requires --ref-kp-dir")
        
        ref_store = stream.BatchedRefStore(
            ref_dir=Path(ref_kp_dir),
            pattern=args.ref_kp_pattern,
            cache_size=args.ref_kp_cache,
            max_kpts=args.se_topk
        )
        print(f"[INFO] Ref kp store: {ref_kp_dir} (CPU Cache -> Batched GPU)")

    # change models to half
    vit = vit.half()
    se_model.half()

    join_stream = torch.cuda.current_stream()

    if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
        vit0 = torch.cuda.Event(True); vit1 = torch.cuda.Event(True)
        ret0 = torch.cuda.Event(True); ret1 = torch.cuda.Event(True)
    if args.method in ["superevent", "eventgem", "eventgem-d"]:
        se0 = torch.cuda.Event(True);  se1 = torch.cuda.Event(True)
    if args.method == "eventgem-d":
        d0 = torch.cuda.Event(True);   d1 = torch.cuda.Event(True)
    if args.method == "lens":
        lens0 = torch.cuda.Event(True); lens1 = torch.cuda.Event(True)
    if args.method == "sparse":
         sparse0 = torch.cuda.Event(True); sparse1 = torch.cuda.Event(True)
    if args.method == "eventvlad":
        vlad0 = torch.cuda.Event(True); vlad1 = torch.cuda.Event(True)
    if args.live_davis:
        davis0 = torch.cuda.Event(True); davis1 = torch.cuda.Event(True)

    j0 = torch.cuda.Event(True);   j1 = torch.cuda.Event(True)

    t_read_list, t_vit_list, t_se_list, t_rerank_list, t_total_list, t_vlad_list = [], [], [], [], [], []
    n_events_list = []

    target_period = 1.0 / max(1e-6, args.target_hz)
    wall0 = time.perf_counter()

    print("[INFO] Starting Loop...", flush=True)

    reranked_cols = []
    sims = []
    queries = []
    ref_feats = []
    with torch.inference_mode():
        if args.live_davis:
            with torch.cuda.stream(stream_davis):
                davis0.record(stream_davis)
                event_iter = stream.stream_event_windows_davis_live(args.dt_ms)
                davis1.record(stream_davis)
        else:
            event_iter = stream.stream_event_windows_raw(
                hdf5_path, args.dt_ms, args.chunk_size, args.time_scale, args.start_time, args.max_frames
            )
        for (_, _, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms) in event_iter:
            cpu0 = time.perf_counter()
            n_events = int(x.size)
            j0.record(join_stream)
            
            if x.size == 0:
                continue

            if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
                with torch.cuda.stream(stream_vit):
                    vit0.record(stream_vit)
                    pol_2hw, _ = stream.gpu_polarity_frame_2ch(x, y, p, H, W, device)
                    inp = stream.vit_preprocess_like_dataloader(pol_2hw, out_hw=(vitH, vitW))
                    inp = inp.to(dtype=torch.float16)
                    if (H != vitH) or (W != vitW):
                        mode = "nearest" if args.resize == "nearest" else "bilinear"
                        inp = F.interpolate(inp, size=(vitH, vitW), mode=mode, align_corners=False if mode=="bilinear" else None)

                    if args.amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            q_desc_vit = stream.vit_gem_descriptor(vit, inp)
                    else:
                        q_desc_vit = stream.vit_gem_descriptor(vit, inp)
                    
                    if args.extract_reference:
                        ref_feats.append(q_desc_vit.cpu().numpy())
                    else:
                        ret0.record(stream_vit)
                        top_idx_t, top_dist_t, sims_t = stream.retrieve_topk(ref_db, q_desc_vit, k=int(args.retrieval_k), return_sims=True)
                        ret1.record(stream_vit)
                        vit1.record(stream_vit)
                        sims.append(sims_t.cpu())

            if args.method in ["superevent", "eventgem", "eventgem-d"]:
                with torch.cuda.stream(stream_se):
                    se0.record(stream_se)
                    mcts = stream.gpu_mcts(x, y, t_raw, p, H, W, int(t_ref_raw), float(args.time_scale), windows_sec, device)
                    mcts = mcts[:, off_top:h_end, off_left:w_end] 
                    
                    pred = se_model(mcts.unsqueeze(0).to(dtype=torch.float16))
                    prob, desc_map = pred['prob'], pred['descriptors']
                    
                    kpts_all, _ = fast_nms(prob, se_cfg, top_k=int(args.se_topk))
                    kpts_yx = kpts_all[0]

                    q_k_desc = stream.sample_descriptors_at_kpts(kpts_yx.float(), desc_map)
                    se1.record(stream_se)

                    if args.extract_reference:
                        ref_kp_path = ref_kp_dir / f"{args.ref_kp_pattern.format(frame_idx)}"
                        np.savez(ref_kp_path, kpts=kpts_yx.cpu().numpy(), desc=q_k_desc.cpu().numpy())
            
            if args.method == "eventgem-d":
                with torch.cuda.stream(stream_d):
                    d0.record(stream_d)
                    tencode = stream.tencode(x, y, t_raw, p, height=H, width=W, white_frame=False, normalize=True, device=device)
                    # Inference (no grad, via decorator)
                    with torch.autocast(device_type="cuda", enabled=True):
                        if model_name == 'DAv2':
                            pred = depth_model.infer_image(tencode.unsqueeze(0))  # (1,1,H,W)
                        elif model_name == 'RecDAv2':
                            pred, prev_states = depth_model.infer_image(tencode.unsqueeze(0), prev_states=prev_states)
                        else:
                            raise ValueError(f"Model {model_name} not implemented in this script.")
                    d1.record(stream_d)
                    if args.extract_reference:
                        depth_path = ref_depth_dir / f"{args.depth_pattern.format(frame_idx)}"
                        # save as png with values scaled to 16-bit range
                        depth_to_save = (pred.squeeze().cpu().numpy() * 65535.0).astype(np.uint16)
                        imageio.imwrite(depth_path, depth_to_save)

            if args.method == "lens":
                with torch.cuda.stream(stream_lens):
                    lens0.record(stream_lens)
                    pol_2hw, _ = stream.gpu_polarity_frame_2ch(x, y, p, H, W, device)
                    # combine polarities by summating
                    pol_combined = pol_2hw.sum(dim=0, keepdim=True)
                    lens_model.evaluate(pol_combined)
                    lens1.record(stream_lens)
            if args.method == "sparse":
                sparse0.record(stream_sparse)
                pol_2hw, _ = stream.gpu_polarity_frame_2ch(x, y, p, H, W, device)
                pol_combined = pol_2hw.sum(dim=0, keepdim=True)
                # sample at random pixels
                sampled_events = pol_combined[:, y_coords, x_coords]
                # run a sum of absolute differences
                stream.get_distance_matrix(sparse_reference_data, sampled_events)
                sparse1.record(stream_sparse)
            if args.method == "eventvlad":
                vlad0.record(join_stream)
                pol_2hw, _ = stream.gpu_polarity_frame_2ch(x, y, p, H, W, device)
                pol_combined = pol_2hw.sum(dim=0).cpu().numpy()
                # we need to wait for 3 inputs to denoise
                if len(queries) < 3:
                    queries.append(pol_combined)
                    continue
                else:
                    queries.append(pol_combined)
                    denoise_query = stream.stream_vlad_denoise(queries, denoise_model)
                    q_desc_vlad = stream.extract_eventvlad_features(netvlad_model, denoise_query)
                    vlad1.record(join_stream)
                    # remove first queries index
                    queries.pop(0)


            if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
                join_stream.wait_stream(stream_vit)
            if args.method in ["superevent", "eventgem", "eventgem-d"]:
                join_stream.wait_stream(stream_se)
            if args.method == "eventgem-d":
                join_stream.wait_stream(stream_d)
            if args.method == "lens":
                join_stream.wait_stream(stream_lens)
            if args.method == "sparse":
                join_stream.wait_stream(stream_sparse)
            if args.method == "eventvlad":
                join_stream.wait_stream(stream_vlad)
            if args.live_davis:
                join_stream.wait_stream(stream_davis)

            j1.record(join_stream)
            


            t_rerank0 = time.perf_counter()

            best_idx = int(top_idx_t[0].item()) if top_idx_t.numel() else -1
            best_inl = 0

            if args.method in ["eventgem", "eventgem-d"]:
                if top_idx_t.numel() > 0 and kpts_yx.numel() > 10:
                    q_xy = kpts_yx[:, [1,0]].float()
                    q_xy[:,0] += float(off_left)
                    q_xy[:,1] += float(off_top)
                    
                    cand_ids = top_idx_t.cpu().numpy().astype(np.int64)
                    cand_dist_val = top_dist_t.cpu().numpy()
                    
                    inlier_counts = stream.batched_ransac_rerank(
                        q_xy, q_k_desc, ref_store, cand_ids, 
                        max_matches=170, ratio_thresh=float(args.match_ratio)
                    )

                    # after you compute inlier_counts for cand_ids, build the new column:
                    new_col = stream.build_reranked_column_from_sims(
                        sims_t=sims_t,
                        cand_ids=cand_ids,
                        inlier_counts=inlier_counts,
                        inlier_weight=float(args.inlier_weight),
                    )
                    # then stash it for later matrix assembly (or write to disk)
                    if args.method == "eventgem":
                        reranked_cols.append(new_col)
                        final_scores = cand_dist_val - (inlier_counts * args.inlier_weight)
                        best_arg = np.argmin(final_scores)
                        best_idx = cand_ids[best_arg]
                        best_inl = inlier_counts[best_arg]

                    if args.method == "eventgem-d":
                        # find the new best candiate
                        reranked = stream.rerank_depth_single_query(qD=pred, base_dist=new_col, ref_depth_dir=args.ref_depth_dir, top_k=args.retrieval_k)
                        reranked_cols.append(reranked)
                        best_idx = int(reranked.argmin().item())
                        best_inl = 0  # not applicable for depth-based re-ranking

                t_rerank = (time.perf_counter() - t_rerank0) * 1000.0
            elif args.method == "superevent":
                # Run brute force matching on superevent points
                dist = []
                dist.append(stream.bruteforce(q_k_desc, ref_descs))

            # End of processing for this frame, record total time
            t_total = (time.perf_counter() - cpu0) * 1000.0
            t_read_list.append(t_read_ms)

            if (frame_idx % 100) == 0:
                torch.cuda.synchronize()
                hz = 1000.0 / max(1e-6, t_total)
                # just print the total time lapsed
                print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} total={t_total:.1f}ms ({hz:.1f} Hz)", flush=True)

    wall_s = time.perf_counter() - wall0
    n_frames = len(t_total_list)

    if args.method == "eventgem":
        print("\n========== SUMMARY ==========")
        print(f"Frames: {n_frames} | Wall: {wall_s:.2f}s | Avg FPS: {n_frames/wall_s:.2f}")
        t_total_np = np.array(t_total_list)
        print(stream.summarize_ms(np.array(t_read_list), "Read"))
        print(stream.summarize_ms(np.array(t_vit_list), "ViT (GPU)"))
        print(stream.summarize_ms(np.array(t_se_list), "SE (GPU)"))
        print(stream.summarize_ms(np.array(t_rerank_list), "Rerank (Batch GPU)"))
        print(stream.summarize_ms(t_total_np, "Total End2End"))
        print(f"Over budget ({args.dt_ms}ms): {np.sum(t_total_np > args.dt_ms)}/{n_frames}")
    elif args.method == "eventgem-d":
        print("\n========== SUMMARY ==========")
        print(f"Frames: {n_frames} | Wall: {wall_s:.2f}s | Avg FPS: {n_frames/wall_s:.2f}")
        t_total_np = np.array(t_total_list)
        print(stream.summarize_ms(np.array(t_read_list), "Read"))
        print(stream.summarize_ms(np.array(t_vit_list), "ViT (GPU)"))
        print(stream.summarize_ms(np.array(t_se_list), "SE (GPU)"))
        print(stream.summarize_ms(np.array(t_rerank_list), "Rerank (Batch GPU)"))
        print(stream.summarize_ms(t_total_np, "Total End2End"))
        print(f"Over budget ({args.dt_ms}ms): {np.sum(t_total_np > args.dt_ms)}/{n_frames}")
    elif args.method == "sparse":
        print("\n========== SUMMARY ==========")
        print(f"Frames: {n_frames} | Wall: {wall_s:.2f}s | Avg FPS: {n_frames/wall_s:.2f}")
        t_total_np = np.array(t_total_list)
        print(stream.summarize_ms(np.array(t_read_list), "Read"))
        print(stream.summarize_ms(np.array(sparse_ms), "Sparse (GPU)"))
        print(stream.summarize_ms(t_total_np, "Total End2End"))
        print(f"Over budget ({args.dt_ms}ms): {np.sum(t_total_np > args.dt_ms)}/{n_frames}")
    else: # print eventvlad summary
        print("\n========== SUMMARY ==========")
        print(f"Frames: {n_frames} | Wall: {wall_s:.2f}s | Avg FPS: {n_frames/wall_s:.2f}")
        t_total_np = np.array(t_total_list)
        print(stream.summarize_ms(np.array(t_read_list), "Read"))
        # vlad ms
        print(stream.summarize_ms(np.array(t_vlad_list), "VLAD (GPU)"))
        print(stream.summarize_ms(np.array(t_rerank_list), "Rerank (Batch GPU)"))
        print(stream.summarize_ms(t_total_np, "Total End2End"))
        print(f"Over budget ({args.dt_ms}ms): {np.sum(t_total_np > args.dt_ms)}/{n_frames}")

    # stack the reranked cols
    if args.gt_file is not None:
        reranked_cols_stack = np.stack(reranked_cols, axis=0)
        reranked_cols_stack = reranked_cols_stack.T
        S_in = 1-reranked_cols_stack

        K = [1, 5, 10]
        # load ground truth file
        gt = np.load(args.gt_file)
        # resize to match shape of reranked_cols_stack if needed
        from skimage.transform import resize
        gt_resized = resize(gt, S_in.shape, order=0, preserve_range=True, anti_aliasing=False)
        from prettytable import PrettyTable
        table = PrettyTable()

        # add columns for each K
        for k in K:
            table.add_column(f"Recall@{k}", [stream.recallAtK(S_in, gt_resized, K=k)])

        print(table)

if __name__ == "__main__":
    main()