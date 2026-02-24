#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import streamutils.stream as stream
import utils.convert_ref as convert_ref
import utils.convert_feats as convert_feats

import sys
import numpy as np
import torch
import ctypes
import torch.nn.functional as F
from tqdm import tqdm

import threading
import sys

def wait_for_enter(prompt: str = ""):
    try:
        return input(prompt)
    except EOFError:
        # if stdin isn't available, don't crash
        return ""

def countdown(seconds: int = 3):
    for i in range(seconds, 0, -1):
        print(f"Starting in {i}...", flush=True)
        time.sleep(1.0)
    print("GO!\n", flush=True)

def start_stop_listener(stop_event: threading.Event):
    # Second Enter: request stop
    wait_for_enter("\nPress ENTER to stop the experiment...\n")
    stop_event.set()

THIS_DIR = Path(__file__).resolve().parent
BACKBONE_ROOT = THIS_DIR / "external" / "backbone"
SUPEREVENT_ROOT = THIS_DIR / "external" / "superevent"

# Order matters: backbone's `utils` must win
sys.path.insert(0, str(SUPEREVENT_ROOT))
sys.path.insert(0, str(BACKBONE_ROOT))

lib = ctypes.CDLL("./libtrt_runner.so")

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
    ap.add_argument("--dataset",  type=str, required=True,
                    help="Name of the dataset")
    ap.add_argument("--reference", type=str, required=True,
                    help="Name of the reference")
    ap.add_argument("--query", type=str, required=True,
                    help="Name of the query")
    ap.add_argument("--time-scale", type=float, default=1e-6,
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
    
    # Evaluation metrics and parameters
    ap.add_argument("--target-hz", type=float, default=20.0)

    args = ap.parse_args()
    
    # If extracting reference information, set storage paths
    # if args.extract_reference:
    ref_feats_dir = Path(args.features_dir) / args.dataset / f"{args.reference}-{args.dt_ms}"
    ref_feats_file = ref_feats_dir / f"{args.dataset}_{args.reference}_features.pt"
    ref_kp_dir = Path(args.keypoint_dir) / args.dataset / f"{args.reference}-{args.dt_ms}"
    qry_kp_dir = Path(args.keypoint_dir) / args.dataset / f"{args.query}-{args.dt_ms}"
    qry_rerank_dir = Path(args.features_dir) / args.dataset / f"{args.query}-{args.dt_ms}" / "rerank"
    ref_depth_dir = Path(args.depth_dir) / args.dataset / f"{args.reference}-{args.dt_ms}"
    ref_feats_dir.mkdir(parents=True, exist_ok=True)
    ref_kp_dir.mkdir(parents=True, exist_ok=True)
    ref_depth_dir.mkdir(parents=True, exist_ok=True)
    qry_feat_dir = Path(args.features_dir) / args.dataset / f"{args.query}-{args.dt_ms}"
    qry_feat_dir.mkdir(parents=True, exist_ok=True)
    qry_rerank_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracted reference features will be saved to: {ref_feats_dir}")
    print(f"[INFO] Extracted reference keypoints will be saved to: {ref_kp_dir}")
    print(f"[INFO] Extracted reference depth maps will be saved to: {ref_depth_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    H, W = 260, 346  # default DAVIS resolution
    stream_davis = torch.cuda.Stream()
    vit = TRTEngineCUDA12("vitgem.engine")
    vitH, vitW = 224, 224
    print(f"[INFO] ViT expects ~ {vitH}x{vitW}")
    if not args.extract_reference:
        ref_db = stream.load_ref_vit_db(ref_feats_file, device=device, dtype=torch.float16 if args.amp else torch.float32)
        ref_db = ref_db.to(dtype=torch.float16)
    stream_vit = torch.cuda.Stream()

    se_model = TRTEngineCUDA12("superevent.engine")
    se_cfg = stream.load_superevent_config(Path(args.se_config))
    from models.util import fast_nms
    off_top, off_left, _, _, h_end, w_end, Hc, Wc = stream.compute_superevent_crop_offsets(H, W, se_cfg)
    print(f"[INFO] SuperEvent crop: top={off_top} left={off_left} -> {Hc}x{Wc}")
    stream_se = torch.cuda.Stream()
    windows_sec = torch.tensor(np.array(args.mcts_windows_ms, dtype=np.float32) * 1e-3, device=device)
    
    stop_event = threading.Event()

    # Start preview
    preview = stream.LiveEventPreview(scale=1.5, stop_event=stop_event)
    preview.start()

    # 1) Enter to start + countdown
    wait_for_enter("Press ENTER to start streaming...\n")
    countdown(seconds=3)

    # 2) Start listener thread for stop
    listener = threading.Thread(target=start_stop_listener, args=(stop_event,), daemon=True)
    listener.start()

    raw_logger = None
    ref_store = None
    
    ref_store = stream.BatchedRefStore(
        ref_dir=Path(ref_kp_dir),
        pattern=args.ref_kp_pattern,
        cache_size=args.ref_kp_cache,
        max_kpts=args.se_topk
    )

    # change models to half

    join_stream = torch.cuda.current_stream()

    vit0 = torch.cuda.Event(True); vit1 = torch.cuda.Event(True)
    ret0 = torch.cuda.Event(True); ret1 = torch.cuda.Event(True)
    se0 = torch.cuda.Event(True);  se1 = torch.cuda.Event(True)
    davis0 = torch.cuda.Event(True); davis1 = torch.cuda.Event(True)
    j0 = torch.cuda.Event(True);   j1 = torch.cuda.Event(True)

    t_read_list, t_vit_list, t_se_list, t_rerank_list, t_total_list, t_vlad_list = [], [], [], [], [], []
    wall0 = time.perf_counter()

    print("[INFO] Starting Loop...", flush=True)
    import dv_processing as dv
    B = dv.io.camera.DAVIS.Davis346BiasCF

    bias_steps_cf = {
        B.On:         (+1, +63),
        B.Off:        (+5, +168),
        B.Refractory: (0, 0),
    }

    event_iter = stream.stream_event_windows_davis_live(
        args.dt_ms,
        on_window=preview.enqueue,
        bias_steps_cf=bias_steps_cf,
    )
    #plotter = stream.LiveDistPlotCV(n_refs=ref_store.num_refs, update_hz=10.0)
    reranked_cols = []
    sims = []
    queries = []
    ref_feats = []
    frame = 0
    try:
        with torch.inference_mode():
            with torch.cuda.stream(stream_davis):
                event_iter = stream.stream_event_windows_davis_live(
                            args.dt_ms,
                            on_window=preview.enqueue,
                            bias_steps_cf=bias_steps_cf
                        )
            for (_, _, t_ref_raw, x, y, t_raw, p, frame_idx, t_read_ms) in event_iter:
                cpu0 = time.perf_counter()
                if stop_event.is_set():
                    print("\n[STOP] Stop requested. Finishing gracefully...", flush=True)
                    break
                x = torch.tensor(x).to(device='cuda')
                y = torch.tensor(y).to(device='cuda')
                p = torch.tensor(p).to(device='cuda')
                t_raw = torch.tensor(t_raw).to(device='cuda')

                j0.record(join_stream)
                if x.size == 0:
                    continue

                with torch.cuda.stream(stream_vit):
                    vit0.record(stream_vit)
                    pol_2hw, _ = stream.gpu_polarity_frame_2ch(x, y, p, H, W, device)
                    inp = stream.vit_preprocess_like_dataloader(pol_2hw, out_hw=(vitH, vitW))
                    inp = inp.to(dtype=torch.float16)
                    if (H != vitH) or (W != vitW):
                        mode = "nearest" if args.resize == "nearest" else "bilinear"
                        inp = F.interpolate(inp, size=(vitH, vitW), mode=mode, align_corners=False if mode=="bilinear" else None)

                    q_desc_vit = vit(inp)
                    
                    if args.extract_reference:
                        if not (frame_idx < 10):
                            np.savez(f"{ref_feats_dir}/ref_feats_{frame_idx}.npz", q_desc_vit.cpu().numpy())
                    else:
                        ret0.record(stream_vit)

                        top_idx_t, top_dist_t, sims_t = stream.retrieve_topk(ref_db, q_desc_vit, k=int(args.retrieval_k), return_sims=True)
                        ret1.record(stream_vit)
                        vit1.record(stream_vit)
                        sims.append(sims_t.cpu())

                with torch.cuda.stream(stream_se):
                    se0.record(stream_se)

                    # MCTS -> crop
                    mcts = stream.gpu_mcts(
                        x, y, t_raw, p, H, W,
                        int(t_ref_raw),
                        float(args.time_scale),      # IMPORTANT: use 1e-6 for DAVIS
                        windows_sec,
                        device
                    )
                    mcts = mcts[:, off_top:h_end, off_left:w_end]  # (C,Hc,Wc)

                    # SuperEvent
                    pred = se_model(mcts.unsqueeze(0).to(dtype=torch.float16))  # (1,C,Hc,Wc) -> outputs
                    prob, desc_map = pred[1], pred[3]   # (adjust if your se_model returns dict)

                    # NMS first (need kpts + scores for saving)
                    kpts_all, scores_all = fast_nms(prob, se_cfg, top_k=int(args.se_topk))

                    kpts_yx = kpts_all[0]          # (N,2) or (N,3)
                    scores  = scores_all[0] if scores_all is not None else None

                    # If kpts are (y,x,score), strip coords and (optionally) use 3rd col as score
                    if kpts_yx.ndim == 3 and kpts_yx.shape[0] == 1:
                        kpts_yx = kpts_yx.squeeze(0)
                    if kpts_yx.shape[-1] > 2:
                        if scores is None:
                            scores = kpts_yx[:, 2]
                        kpts_yx = kpts_yx[:, :2]   # keep (y,x) only

                    # Sample descriptors once (works for both saving and query)
                    desc_sampled = stream.sample_descriptors_at_kpts(kpts_yx.float(), desc_map)  # (N,D)

                    if args.extract_reference:
                        if raw_logger is None:
                            D = int(desc_sampled.shape[1])
                            raw_logger = stream.RawKPLogger(
                                Path(ref_kp_dir) / "ref_kp_raw.bin",
                                H=H, W=W,
                                off_top=off_top, off_left=off_left,
                                top_k=int(args.se_topk),
                                D=D,
                            )
                        # scores can be None; your logger expects a tensor -> make a zeros tensor
                        if scores is None:
                            scores = torch.zeros((kpts_yx.shape[0],), device=desc_sampled.device, dtype=torch.float32)

                        raw_logger.write(frame_idx, int(t_ref_raw), kpts_yx, scores, desc_sampled)

                    else:
                        # Use the same sampled descriptors for your normal path
                        q_k_desc = desc_sampled
                        if raw_logger is None:
                            D = int(desc_sampled.shape[1])
                            raw_logger = stream.RawKPLogger(
                                Path(qry_kp_dir) / "ref_kp_raw.bin",
                                H=H, W=W,
                                off_top=off_top, off_left=off_left,
                                top_k=int(args.se_topk),
                                D=D,
                            )
                        # scores can be None; your logger expects a tensor -> make a zeros tensor
                        if scores is None:
                            scores = torch.zeros((kpts_yx.shape[0],), device=desc_sampled.device, dtype=torch.float32)

                        raw_logger.write(frame_idx, int(t_ref_raw), kpts_yx, scores, desc_sampled)

                    se1.record(stream_se)
            
                if not args.extract_reference:
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
                        final_scores = cand_dist_val - (inlier_counts * args.inlier_weight)
                        new_dist = stream.build_reranked_column_from_sims(sims_t, cand_ids, inlier_counts, args.inlier_weight)
                        #plotter.update(new_dist)
                        if not (frame_idx < 10):
                            np.savez_compressed(f"{qry_feat_dir}/ref_feats_{frame_idx}.npz", q_desc_vit.cpu().numpy())
                            np.savez_compressed(f"{qry_rerank_dir}/rerank_{frame_idx}.npz", new_dist)

                # End of processing for this frame, record total time
                t_total = (time.perf_counter() - cpu0) * 1000.0
                t_read_list.append(t_read_ms)

                if (frame_idx % 100) == 0:
                    hz = 1000.0 / max(1e-6, t_total)
                    # just print the total time lapsed
                    print(f"[LIVE] {frame_idx:5d} total={t_total:.1f}ms ({hz:.1f} Hz)", flush=True)

    finally:
        wall_s = time.perf_counter() - wall0
        n_frames = len(t_total_list)
        preview.stop()

        # convert the keypoints to the format expected by the reference (for evaluation)
        if args.extract_reference:
            convert_ref.main(
                raw_path=str(Path(ref_kp_dir) / "ref_kp_raw.bin"),
                out_dir=str(Path(ref_kp_dir) / f"kps_{args.reference}")
            )
            convert_feats.main(
                npz_dir=str(Path(ref_feats_dir)),
                out=str(Path(ref_feats_file))
            )
        else:
            convert_feats.main(
                npz_dir=str(Path(qry_feat_dir)),
                out=str(Path(qry_feat_dir) / f"{args.dataset}_{args.query}_features.pt")
            )

if __name__ == "__main__":
    main()
