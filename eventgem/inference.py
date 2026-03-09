#!/usr/bin/env python3
import gzip
import time
from pathlib import Path
import eventgem.streamutils.stream as stream
import os
import eventgem.streamutils.convert_ref as convert_ref
import tarfile 
import requests
import shutil

import sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from eventgem.external.depthanyevent.models import fetch_model  # from the DepthAnyEvent repo
import imageio
import matplotlib.pyplot as plt
THIS_DIR = Path(__file__).resolve().parent
BACKBONE_ROOT = THIS_DIR / "external" / "backbone"
SUPEREVENT_ROOT = THIS_DIR / "external" / "superevent"

# Order matters: backbone's `utils` must win
sys.path.insert(0, str(SUPEREVENT_ROOT))
sys.path.insert(0, str(BACKBONE_ROOT))

# ---------------------------
# Main
# ---------------------------
def stream_file(args):

    # If extracting features, set sequence to reference
    sequence = args.reference if args.extract_reference else args.query

    hdf5_path = Path(f"{args.data_root}/{args.dataset}/{sequence}/{sequence}.hdf5")
    # File path for hdf5
    if not args.live_davis:
        if not hdf5_path.exists():
            raise FileNotFoundError(hdf5_path)
    
    # If extracting reference information, set storage paths
    # if args.extract_reference:
    ref_feats_dir = Path(args.feature_out) / args.dataset / f"{args.reference}-{args.dt_ms}"
    ref_feats_file = ref_feats_dir / f"{args.dataset}_{args.reference}_features.pt"

    ref_kp_dir = Path(args.keypoint_out) / args.dataset / f"{args.reference}-{args.dt_ms}" / f"kps_{args.reference}"
    ref_depth_dir = Path(args.depth_out) / args.dataset / f"{args.reference}-{args.dt_ms}"
    ref_feats_dir.mkdir(parents=True, exist_ok=True)
    if not args.demo:
        ref_kp_dir.mkdir(parents=True, exist_ok=True)
    ref_depth_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracted reference features will be saved to: {ref_feats_dir}")
    print(f"[INFO] Extracted reference keypoints will be saved to: {ref_kp_dir}")
    print(f"[INFO] Extracted reference depth maps will be saved to: {ref_depth_dir}")
    gt_path = f"{args.data_root}/{args.dataset}/ground_truth/{args.reference}_{args.query}_GT.npy"
    # If running the demo, download the reference features and kps
    if args.demo:
        feat_link = "https://huggingface.co/datasets/AdamHines/EventGeM/resolve/main/brisbane_event_sunset2_features.pt"
        kps_link = "https://huggingface.co/datasets/AdamHines/EventGeM/resolve/main/kps_sunset2.tar.gz"
        gt_link = "https://huggingface.co/datasets/AdamHines/EventGeM/resolve/main/sunset2_sunset1_GT.npy"

        args.kp_pattern = "ref_kp_{:05d}.npz"
        # download reference features to the ref_feats_file location
        if not ref_feats_file.exists():
            print(f"[INFO] Downloading reference features from {feat_link}...")
            response = requests.get(feat_link)
            with open(ref_feats_file, "wb") as f:
                f.write(response.content)
        # download and extract reference keypoints to the ref_kp_dir location
        if not ref_kp_dir.exists():
            ref_kp_dir.mkdir(parents=True, exist_ok=True)

            archive_path = ref_kp_dir / "kps_sunset2.tar.gz"

            print(f"[INFO] Downloading {kps_link} -> {archive_path}")

            with requests.get(
                kps_link,
                stream=True,
                allow_redirects=True,
                timeout=120,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as r:
                r.raise_for_status()

                with open(archive_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            print(f"[INFO] Downloaded size: {archive_path.stat().st_size} bytes")

            with gzip.open(archive_path, "rb") as f:
                f.read(1)

            print("[INFO] Valid gzip file")

            if not tarfile.is_tarfile(archive_path):
                raise RuntimeError(f"Downloaded file is not a valid tar archive: {archive_path}")

            extracted = 0
            with tarfile.open(archive_path, "r:gz") as tar:
                members = tar.getmembers()
                print(f"[INFO] Archive contents (first 10): {[m.name for m in members[:10]]}")

                for member in members:
                    if not member.isfile():
                        continue
                    if not member.name.endswith(".npz"):
                        continue

                    out_path = ref_kp_dir / Path(member.name).name  # strip all directories

                    src = tar.extractfile(member)
                    if src is None:
                        continue

                    with src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)

                    extracted += 1

            print(f"[INFO] Extracted {extracted} .npz files directly to {ref_kp_dir}")

        # download GT file to the dataset output directory
        if not Path(gt_path).exists():
            # make the ground truth directory if it doesn't exist
            gt_dir = Path(gt_path).parent
            gt_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Downloading GT file from {gt_link}...")
            response = requests.get(gt_link)
            with open(gt_path, "wb") as f:
                f.write(response.content)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if not args.live_davis:
        with h5py.File(hdf5_path, "r") as f:
            x_dset, y_dset, t_dset, p_dset = stream.find_event_datasets(f)
            H, W = stream.infer_resolution(x_dset, y_dset, args.chunk_size)
            
            t0_raw = int(t_dset[0]); tN_raw = int(t_dset[-1])
            print(f"[INFO] Sensor: {H}x{W}")
    else:
        H, W = 260, 346  # default DAVIS resolution
        stream_davis = torch.cuda.Stream()

    print(args.method)
    # Load the corresponding model to run the inference on
    if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
        vit = stream.load_vit_backbone(args.backbone_ckpt, device)
        vitH, vitW = 224, 224
        print(f"[INFO] ViT expects ~ {vitH}x{vitW}")
        if not args.extract_reference:
            ref_db = stream.load_ref_vit_db(ref_feats_file, device=device, dtype=torch.float32)
        stream_vit = torch.cuda.Stream()

    if args.method in ["superevent", "eventgem", "eventgem-d"]:
        se_model, se_cfg = stream.build_superevent_model(Path(args.se_config), Path(args.se_weights), device)
        from models.util import fast_nms
        off_top, off_left, _, _, h_end, w_end, Hc, Wc = stream.compute_superevent_crop_offsets(H, W, se_cfg)
        print(f"[INFO] SuperEvent crop: top={off_top} left={off_left} -> {Hc}x{Wc}")
        stream_se = torch.cuda.Stream()
        windows_sec = torch.tensor(np.array(args.mcts_time, dtype=np.float32) * 1e-3, device=device)
        if args.method == "superevent":
            # Loading the reference keypoint descriptors
            ref_descs = stream.preload_ref_descs(ref_kp_dir, args.kp_pattern)

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
            pattern=args.kp_pattern,
            cache_size=args.ref_kp_cache,
            max_kpts=args.se_topk,
        )
    raw_logger = None

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
    # Pre-load your reference descriptors to GPU
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
                hdf5_path, args.dt_ms, args.chunk_size, args.time_scale, args.start_time, args.skip
            )

        new_cols = []
        for (_, _, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms) in event_iter:
            cpu0 = time.perf_counter()
            n_events = int(x.size)
            j0.record(join_stream)

            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            t_raw = torch.from_numpy(t_raw).to(device)
            p = torch.from_numpy(p).to(device)

            if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
                with torch.cuda.stream(stream_vit):
                    vit0.record(stream_vit)
                    pol_2hw, _ = stream.gpu_polarity_frame_2ch(x, y, p, H, W, device)
                    inp = stream.vit_preprocess_like_dataloader(pol_2hw, out_hw=(vitH, vitW))
                    if (H != vitH) or (W != vitW):
                        mode = "nearest"
                        inp = F.interpolate(inp, size=(vitH, vitW), mode=mode, align_corners=False if mode=="bilinear" else None)
                        q_desc_vit = stream.vit_gem_descriptor(vit, inp)

                    if args.extract_reference:
                        ref_feats.append(q_desc_vit.cpu().numpy())
                    else:
                        ret0.record(stream_vit)
                        
                        top_idx_t, top_dist_t, sims_t = stream.retrieve_topk(ref_db, q_desc_vit, k=int(args.top_k), return_sims=True)

                        ret1.record(stream_vit)
                        vit1.record(stream_vit)

            if args.method in ["superevent", "eventgem", "eventgem-d"]:
                with torch.cuda.stream(stream_se):
                    se0.record(stream_se)
                    mcts = stream.gpu_mcts(x, y, t_raw, p, H, W, int(t_ref_raw), float(args.time_scale), windows_sec, device)
                    mcts = mcts[:, off_top:h_end, off_left:w_end] 

                    pred = se_model(mcts.unsqueeze(0))

                    prob, desc_map = pred['prob'], pred['descriptors']

                    kpts_all, scores_all = fast_nms(prob, se_cfg, top_k=int(args.se_topk))
                    kpts_yx = kpts_all[0]

                    q_k_desc = stream.sample_descriptors_at_kpts(kpts_yx.float(), desc_map)
                    se1.record(stream_se)

                    scores = scores_all[0]

                    if args.extract_reference:
                        if raw_logger is None:
                            D = int(q_k_desc.shape[1])
                            raw_logger = stream.RawKPLogger(
                                Path(ref_kp_dir) / "ref_kp_raw.bin",
                                H=H, W=W,
                                off_top=off_top, off_left=off_left,
                                top_k=int(args.se_topk),
                                D=D,
                            )
                        # scores can be None; your logger expects a tensor -> make a zeros tensor
                        if scores is None:
                            scores = torch.zeros((kpts_yx.shape[0],), device=q_k_desc.device, dtype=torch.float32)

                        raw_logger.write(frame_idx, int(t_ref_raw), kpts_yx, scores, q_k_desc)

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
            if args.extract_reference:
                continue  # skip the rest of the loop if we're only extracting reference information:
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
            torch.cuda.synchronize()
            
            if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
                vit_ms = vit0.elapsed_time(vit1)
            if args.method in ["superevent", "eventgem", "eventgem-d"]:
                se_ms = se0.elapsed_time(se1)
            if args.method == "eventgem-d":
                d_ms = d0.elapsed_time(d1)
            if args.method == "lens":
                lens_ms = lens0.elapsed_time(lens1)
            if args.method == "sparse":
                sparse_ms = sparse0.elapsed_time(sparse1)
            if args.method == "eventvlad":
                vlad_ms = vlad0.elapsed_time(vlad1)
            if args.live_davis:
                davis_ms = davis0.elapsed_time(davis1)

            t_rerank0 = time.perf_counter()
            if args.method in ["eventgem","eventgem-d"]:
                best_idx = int(top_idx_t[0].item()) if top_idx_t.numel() else -1
                best_inl = 0

            if args.method in ["eventgem", "eventgem-d"]:
                if top_idx_t.numel() > 0 and kpts_yx.numel() > 10:
                    # --- shortlist from global retrieval ---
                    cand_ids = top_idx_t.detach().cpu().numpy().astype(np.int64)             # [K]
                    cand_dist_val = top_dist_t.detach().cpu().numpy().astype(np.float32)    # [K]

                    # --- keypoint rerank via USAC_FAST (returns inliers aligned with cand_ids) ---
                    # inl = stream.usac_fast_inliers_from_refstore(
                    #     q_kpts_yx_t=kpts_yx,
                    #     q_desc_t=q_k_desc,
                    #     ref_store=ref_store,
                    #     cand_ids=cand_ids,
                    #     off_left=off_left,
                    #     off_top=off_top,
                    #     ransac_thresh=float(args.ransac_thresh),
                    #     ratio=0.8,
                    #     max_iters=128,
                    # ).astype(np.int32)    
                    
                    inl = stream.batched_ransac_rerank(
                            kpts_yx, q_k_desc, ref_store, cand_ids, off_left, off_top,
                            max_matches=170, ratio_thresh=float(args.match_ratio)
                        )                                    # [K]

                    kp_new_dists = cand_dist_val - inl.astype(np.float32) * float(args.inlier_weight)  # [K]

                    # --- method-specific handling ---
                    if args.method == "eventgem":
                        new_col = stream.build_reranked_column_from_sims(sims_t, cand_ids, inl, args.inlier_weight)


                        new_cols.append(new_col)

                        best_j = int(np.argmin(kp_new_dists))
                        best_idx = int(cand_ids[best_j])
                        best_inl = int(inl[best_j])

                    elif args.method == "eventgem-d":
                        # Depth rerank expects a full (R,) vector indexed by real ref ids.
                        # Build a temporary base vector: only topK are finite, rest are +inf.
                        # This is cheap and avoids touching the rest of the database.
                        R = int(ref_db.shape[0])  # or whatever holds your number of refs
                        base_full = np.full((R,), np.inf, dtype=np.float32)
                        base_full[cand_ids] = kp_new_dists

                        depth_full = stream.rerank_depth_single_query(
                            qD=pred,                           # your torch depth prediction (1,1,H,W)
                            base_dist=base_full,               # (R,)
                            ref_depth_dir=args.depth_dir,
                            top_k=int(args.retrieval_k),
                            topk_idx=cand_ids,                 # IMPORTANT: only load these depths
                            depth_pattern=getattr(args, "depth_pattern", "depth_{:06d}.png"),
                            depth_index_offset=getattr(args, "depth_index_offset", 0),
                            depth_down_hw=getattr(args, "depth_down_hw", (28, 28)),
                            depth_weight=getattr(args, "depth_weight", 0.15),
                            tau=getattr(args, "depth_tau", 0.3),
                            err_threshold=getattr(args, "depth_err_threshold", 0.7),
                            cache=depth_cache if "depth_cache" in globals() else None,
                        )

                        depth_new_dists = depth_full[cand_ids]   # shortlist only (same order as cand_ids)
                        reranked_cols.append((cand_ids, depth_new_dists))  # sparse stash for later rebuild

                        best_j = int(np.argmin(depth_new_dists))
                        best_idx = int(cand_ids[best_j])
                        best_inl = 0  # depth stage doesn’t use inliers

                t_rerank = (time.perf_counter() - t_rerank0) * 1000.0
            elif args.method == "superevent":
                # Run brute force matching on superevent points
                dist = []
                dist.append(stream.bruteforce(q_k_desc, ref_descs))

            # End of processing for this frame, record total time
            t_total = (time.perf_counter() - cpu0) * 1000.0

            t_read_list.append(t_read_ms)
            if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
                t_vit_list.append(vit_ms)
            if args.method in ["superevent", "eventgem", "eventgem-d"]:
                t_se_list.append(se_ms)
            if args.method == "eventvlad":
                t_vlad_list.append(vlad_ms)
            if args.method in ["eventgem", "eventgem-d"]:
                t_rerank_list.append(t_rerank)
            t_total_list.append(t_total)
            n_events_list.append(n_events)

            if (frame_idx % 100) == 0:
                hz = 1000.0 / max(1e-6, t_total)
                if args.live_davis: # print davis information and vit, se, rerank times
                    print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} davis={davis_ms:.1f}ms ({hz:.1f} Hz) vit={vit_ms:.1f}ms se={se_ms:.1f}ms rerank={t_rerank:.1f}ms total={t_total:.1f}ms", flush=True)
                elif args.method == "eventgem":
                    print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} vit={vit_ms:.1f} se={se_ms:.1f} rerank={t_rerank:.1f} total={t_total:.1f}ms ({hz:.1f} Hz) best={best_idx} inl={best_inl}", flush=True)
                elif args.method == "eventgem-d":
                    print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} vit={vit_ms:.1f} se={se_ms:.1f} d={d_ms:.1f} rerank={t_rerank:.1f} total={t_total:.1f}ms ({hz:.1f} Hz) best={best_idx} inl={best_inl}", flush=True)
                elif args.method == "superevent":
                    print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} se={se_ms:.1f} total={t_total:.1f}ms ({hz:.1f} Hz)", flush=True)
                elif args.method == "lens":
                    print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} lens={lens_ms:.1f} total={t_total:.1f}ms ({hz:.1f} Hz)", flush=True)
                elif args.method == "sparse":
                    print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} sparse={sparse_ms:.1f} total={t_total:.1f}ms ({hz:.1f} Hz)", flush=True)
                else:
                    print(f"[LIVE] {frame_idx:5d} ev={n_events:5d} vlad={vlad_ms:.1f} total={t_total:.1f}ms ({hz:.1f} Hz)", flush=True)

    wall_s = time.perf_counter() - wall0

    n_frames = len(t_total_list)

    if args.extract_reference:
        # normalize features and save as .pt
        ref_feats = np.concatenate(ref_feats, axis=0)
        ref_feats = torch.from_numpy(ref_feats).to(device)
        ref_feats = F.normalize(ref_feats, p=2, dim=1)
        torch.save(ref_feats.cpu(), ref_feats_file)
        print(f"[INFO] Extracted reference features saved to: {ref_feats_file}")
        convert_ref.main(
                raw_path=str(Path(ref_kp_dir) / "ref_kp_raw.bin"),
                out_dir=str(Path(ref_kp_dir) / f"kps_{args.reference}")
            )


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
    if os.path.exists(gt_path):
        # make a big matrix of shape (n_frames, R) where each row is the reranked distance vector for that query frame
        S_in = np.stack(new_cols, axis=0)  # shape (n_frames, R)
        S_in = 1 - S_in.T

        K = [1, 5, 10]
        # load ground truth file
        gt = np.load(gt_path)

        # resize to match shape of reranked_cols_stack if needed
        from skimage.transform import resize
        gt_resized = resize(gt, S_in.shape, order=0, preserve_range=True, anti_aliasing=False)
        from prettytable import PrettyTable
        table = PrettyTable()

        # add columns for each K
        for k in K:
            table.add_column(f"Recall@{k}", [stream.recallAtK(S_in, gt_resized, K=k)])

        print(table)