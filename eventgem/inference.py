#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import streamutils.stream as stream
import os
import streamutils.convert_ref as convert_ref

import ctypes


import sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from external.depthanyevent.models import fetch_model  # from the DepthAnyEvent repo
import imageio
import matplotlib.pyplot as plt
THIS_DIR = Path(__file__).resolve().parent
BACKBONE_ROOT = THIS_DIR / "external" / "backbone"
SUPEREVENT_ROOT = THIS_DIR / "external" / "superevent"

# Order matters: backbone's `utils` must win
sys.path.insert(0, str(SUPEREVENT_ROOT))
sys.path.insert(0, str(BACKBONE_ROOT))

lib = ctypes.CDLL("/home/adam/repo/Event-GeM/libtrt_runner.so")

class TrtHandle(ctypes.Structure):
    pass

lib.trt_load.restype = ctypes.POINTER(TrtHandle)
lib.trt_load.argtypes = [ctypes.c_char_p]

lib.trt_num_io.restype = ctypes.c_int
lib.trt_num_io.argtypes = [ctypes.POINTER(TrtHandle)]

lib.trt_is_input.restype = ctypes.c_int
lib.trt_is_input.argtypes = [ctypes.POINTER(TrtHandle), ctypes.c_int]

lib.trt_get_name.restype = ctypes.c_int
lib.trt_get_name.argtypes = [ctypes.POINTER(TrtHandle), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

lib.trt_get_dtype.restype = ctypes.c_int
lib.trt_get_dtype.argtypes = [ctypes.POINTER(TrtHandle), ctypes.c_int]

lib.trt_get_ndim.restype = ctypes.c_int
lib.trt_get_ndim.argtypes = [ctypes.POINTER(TrtHandle), ctypes.c_int]

lib.trt_get_dims.restype = ctypes.c_int
lib.trt_get_dims.argtypes = [ctypes.POINTER(TrtHandle), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

lib.trt_set_ptr.restype = ctypes.c_int
lib.trt_set_ptr.argtypes = [ctypes.POINTER(TrtHandle), ctypes.c_int, ctypes.c_void_p]

lib.trt_set_stream.argtypes = [ctypes.POINTER(TrtHandle), ctypes.c_ulonglong]

lib.trt_execute.restype = ctypes.c_int
lib.trt_execute.argtypes = [ctypes.POINTER(TrtHandle)]

lib.trt_destroy.argtypes = [ctypes.POINTER(TrtHandle)]

# TRT DataType enum -> torch dtype (common cases)
TRT_TO_TORCH = {
    0: torch.float32,  # kFLOAT
    1: torch.float16,  # kHALF
    2: torch.int8,     # kINT8
    3: torch.int32,    # kINT32
    4: torch.bool,     # kBOOL
}

class TRTEngineCUDA12:
    def __init__(self, engine_path: str):
        self.h = lib.trt_load(engine_path.encode())
        if not self.h:
            raise RuntimeError("Failed to load engine (version mismatch?)")

        self.n = lib.trt_num_io(self.h)
        self.inputs = []
        self.outputs = []
        self._i_idx = []
        self._o_idx = []
        

        # Static-shape allocation (throws if any dim is -1)
        for i in range(self.n):
            dt = lib.trt_get_dtype(self.h, i)
            if dt not in TRT_TO_TORCH:
                raise RuntimeError(f"Unsupported TRT dtype enum {dt} at io[{i}]")
            torch_dtype = TRT_TO_TORCH[dt]

            ndim = lib.trt_get_ndim(self.h, i)
            buf = (ctypes.c_int * 16)()
            ok = lib.trt_get_dims(self.h, i, buf, 16)
            if not ok:
                raise RuntimeError(f"Failed reading dims for io[{i}]")
            shape = [buf[k] for k in range(ndim)]
            if any(s < 0 for s in shape):
                raise RuntimeError(f"io[{i}] is dynamic shape {shape}; handle dynamic separately")

            t = torch.empty(tuple(shape), device="cuda", dtype=torch_dtype)
            ok = lib.trt_set_ptr(self.h, i, ctypes.c_void_p(t.data_ptr()))
            if not ok:
                raise RuntimeError(f"Failed setTensorAddress for io[{i}]")

            if lib.trt_is_input(self.h, i):
                self.inputs.append(t); self._i_idx.append(i)
            else:
                self.outputs.append(t); self._o_idx.append(i)

    def __call__(self, *args):
        for k, a in enumerate(args):
            self.inputs[k].copy_(a)

        # run on the current PyTorch CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        lib.trt_set_stream(self.h, int(stream))

        ok = lib.trt_execute(self.h)
        if ok != 1:
            raise RuntimeError("enqueueV3 failed")
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

    def close(self):
        if self.h:
            lib.trt_destroy(self.h)
            self.h = None

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    # Dataset parameters
    ap.add_argument("--hdf5", type=str, required=True, 
                    help="Path to the input hdf5 file containing events")
    ap.add_argument("--dataset", type=str, required=True,
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
    ap.add_argument('--features-dir', type=str, default='features',
                    help="Directory to save extracted features (if --extract-reference is set)")
    ap.add_argument('--keypoint-dir', type=str, default='keypoints',
                    help="Directory to save extracted keypoints (if --extract-reference is set)")
    ap.add_argument('--depth-dir', type=str, default='depth',
                    help="Directory to save extracted depth maps (if --extract-reference is set)")
    ap.add_argument('--onnx', action='store_true',
                    help="Whether to use ONNX Runtime for ViT inference (instead of PyTorch, GPU only)")
    
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
    ap.add_argument("--retrieval-k", type=int, default=50,
                    help="Number of top candidates to retrieve from ViT before re-ranking")
    ap.add_argument("--se-config", type=str, default="eventgem/external/superevent/config/super_event.yaml",
                    help="Path to the SuperEvent config file")
    ap.add_argument("--se-weights", type=str, default="eventgem/external/superevent/saved_models/super_event_weights.pth",
                    help="Path to the SuperEvent weights file")
    ap.add_argument("--se-topk", type=int, default=170,
                    help="Number of top candidates to keep after re-ranking")
    ap.add_argument("--mcts-windows-ms", type=float, nargs="+", default=[10, 20, 30, 40, 50],
                    help="List of time windows (in ms) for MCTS")
    ap.add_argument("--ref-kp-pattern", type=str, default="ref_kp_{:05d}.npz",
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
    print(ref_feats_file)
    ref_kp_dir = Path(args.keypoint_dir) / args.dataset / f"{args.reference}-{args.dt_ms}" / f"kps_{args.reference}"
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
        if args.onnx:
            vit = TRTEngineCUDA12("/home/adam/repo/Event-GeM/vit.engine")
        else:
            vit = stream.load_vit_backbone(args.backbone_ckpt, device)
        vitH, vitW = 224, 224
        print(f"[INFO] ViT expects ~ {vitH}x{vitW}")
        if not args.extract_reference:
            ref_db = stream.load_ref_vit_db(ref_feats_file, device=device, dtype=torch.float16 if args.amp else torch.float32)
        stream_vit = torch.cuda.Stream()

    if args.method in ["superevent", "eventgem", "eventgem-d"]:
        if args.onnx:
            se_model = TRTEngineCUDA12("/home/adam/repo/Event-GeM/superevent.engine")
            _, se_cfg = stream.build_superevent_model(Path(args.se_config), Path(args.se_weights), device)
        else:
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
    gate = stream.RerankGate(ckpt_dir="/home/adam/repo/Event-GeM/morning", device=device)
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
                hdf5_path, args.dt_ms, args.chunk_size, args.time_scale, args.start_time, args.max_frames
            )
            # -----------------------------
        # Gate/Rerank output stash for evaluation
        # Stores final (cand_ids, cand_dists) for EVERY query frame
        # -----------------------------
        stash_topk = (not args.extract_reference) and (args.method in ["eventgem", "eventgem-d"])
        cand_ids_all: list[np.ndarray] = []
        cand_dists_all: list[np.ndarray] = []
        frame_ids_all: list[int] = []
        rerank_mask_all: list[bool] = []
        K_eval = int(args.retrieval_k)
        new_cols = []
        for (_, _, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms) in event_iter:
            cpu0 = time.perf_counter()
            n_events = int(x.size)
            j0.record(join_stream)

            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            t_raw = torch.from_numpy(t_raw).to(device)
            p = torch.from_numpy(p).to(device)
            
            # if x.size == 0:
            #     continue

            if args.method in ["ecdpt", "eventgem", "eventgem-d"]:
                with torch.cuda.stream(stream_vit):
                    vit0.record(stream_vit)
                    pol_2hw, _ = stream.gpu_polarity_frame_2ch(x, y, p, H, W, device)
                    inp = stream.vit_preprocess_like_dataloader(pol_2hw, out_hw=(vitH, vitW))
                    if (H != vitH) or (W != vitW):
                        mode = "nearest" if args.resize == "nearest" else "bilinear"
                        inp = F.interpolate(inp, size=(vitH, vitW), mode=mode, align_corners=False if mode=="bilinear" else None)
                        if args.onnx:
                            q_desc_vit = vit(inp)
                        else:
                            q_desc_vit = stream.vit_gem_descriptor(vit, inp)

                    if args.extract_reference:
                        ref_feats.append(q_desc_vit.cpu().numpy())
                    else:
                        ret0.record(stream_vit)
                        
                        top_idx_t, top_dist_t, sims_t = stream.retrieve_topk(ref_db, q_desc_vit, k=int(args.retrieval_k), return_sims=True)

                        ret1.record(stream_vit)
                        vit1.record(stream_vit)

            if args.method in ["superevent", "eventgem", "eventgem-d"]:
                with torch.cuda.stream(stream_se):
                    se0.record(stream_se)
                    mcts = stream.gpu_mcts(x, y, t_raw, p, H, W, int(t_ref_raw), float(args.time_scale), windows_sec, device)
                    mcts = mcts[:, off_top:h_end, off_left:w_end] 

                    pred = se_model(mcts.unsqueeze(0))
                    if args.onnx:
                        prob, desc_map = pred[1], pred[3]
                    else:
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
    if args.gt_file is not None:
        # make a big matrix of shape (n_frames, R) where each row is the reranked distance vector for that query frame
        S_in = np.stack(new_cols, axis=0)  # shape (n_frames, R)
        S_in = 1 - S_in.T
        # plt.imshow(S_in, aspect='auto')
        # plt.colorbar()
        # plt.show()
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