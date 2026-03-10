import os
import argparse
import numpy as np
from eventgem.analysis import recall
from eventgem.inference import stream
from eventgem.feature_extraction import EventGeM

def main():
    parser = argparse.ArgumentParser(description="Event-GeM: Pre-trained feature extraction and 2D-homology re-reanking for visual place recognition")
    
    # Dataset parameters
    parser.add_argument("--dataset", "-d", type=str,  
                            choices=["brisbane_event", "nsavp", "fast_slow"],
                            help="Dataset to use for evaluation")
    parser.add_argument("--reference", "-r", type=str, 
                            help="Reference directory to use for evaluation")
    parser.add_argument("--query", "-q", type=str, 
                            help="Query directory to use for evaluation")
    parser.add_argument("--dt-ms", type=int, default=50,
                            help="Reconstruction time window in milliseconds for event datasets")
    parser.add_argument("--mcts-time", type=float, nargs='+', default=[10, 20, 30, 40, 50],
                            help="Space-separated list of temporal window sizes in msec.")
    parser.add_argument("--data-root", type=str, default="./eventgem/data", 
                            help="Root directory for datasets")
    parser.add_argument("--kp-pattern", type=str, default="kp_{:05d}.feat.npz",
                            help="File pattern for keypoint features")
    parser.add_argument("--depth-pattern", type=str, default="depth_{:05d}.png",
                            help="File pattern for depth maps")
    parser.add_argument("--stream", action="store_true",
                            help="Whether to use streaming inference (for large datasets that don't fit in memory)")
    parser.add_argument("--chunk-size", type=int, default=250_000,
                            help="Chunk size for streaming inference")
    parser.add_argument("--time-scale", type=float, default=1e-9,
                            help="Time scale for event accumulation")
    parser.add_argument("--ref-kp-cache", type=int, default=2048,
                            help="Size of the cache for reference keypoints")
    parser.add_argument("--start-time", type=float, default=None,
                             help="Start time for event processing (in seconds)")
    parser.add_argument("--skip", type=int, default=0,
                            help="Number of initial frames to skip (for streaming)")

    # Model parameters
    parser.add_argument("--backbone-ckpt", type=str, default="./eventgem/ckpt/pr.pt",
                            help="Path to the backbone checkpoint")
    parser.add_argument("--se-config", type=str, default="eventgem/external/superevent/config/super_event.yaml",
                    help="Path to the SuperEvent config file")
    parser.add_argument("--se-weights", type=str, default="eventgem/external/superevent/saved_models/super_event_weights.pth",
                    help="Path to the SuperEvent weights file")
    parser.add_argument('--depth-model', default='eventgem/external/depthanyevent/models/rec_dav2/synth/synth.pth', 
                        help='Path to model checkpoint')
    parser.add_argument('--depth-config', type=str, default='eventgem/external/depthanyevent/configs/test/recdav2/rec_dav2_mvsec_test.json',
                        help='Path to config file. If not specified, config from model folder/checkpoint is used')
    parser.add_argument("--top-k", type=int, default=50,
                            help="Number of top candidates to re-rank using 2D-homography")
    parser.add_argument("--se-topk", type=int, default=170,
                            help="Number of top candidates to keep after re-ranking")
    parser.add_argument("--ransac-thresh", type=float, default=5.0, 
                            help="RANSAC pixel threshold (e.g. 3-5 px)")
    parser.add_argument("--inlier-weight", type=float, default=0.05, 
                            help="Distance subtraction per inlier")
    parser.add_argument("--match-ratio", type=float, default=0.8,
                            help="Match ratio for keypoint matching")
    parser.add_argument("--backbone-batch-size", type=int, default=32,
                            help="Batch size for feature extraction")
    parser.add_argument("--keypoint-batch-size", type=int, default=16,
                            help="Batch size for feature extraction")
    parser.add_argument("--feature-out", type=str, default="./eventgem/features",
                            help="Directory to save extracted features")
    parser.add_argument("--keypoint-out", type=str, default="./eventgem/keypoints",
                            help="Directory to save detected keypoints")
    parser.add_argument("--depth-out", type=str, default="./eventgem/depth",
                            help="Directory to save predicted depth maps")
    parser.add_argument("--rerank_mode", type=str, default="keypoints", choices=["keypoints", "depth", "both"],
                            help="Which results to evaluate: original, reranked, or both")
    parser.add_argument("--method", type=str,  default="eventgem", 
                choices=["eventgem", "eventgem-d", "ecdpt", "superevent", "lens", "sparse", "eventvlad"],
                help="Which method to run (for ablation or comparison)")
    
    # Re-run options
    parser.add_argument("--rerun-features", action="store_true",
                            help="Whether to re-run feature extraction even if features already exist")
    parser.add_argument("--rerun-keypoints", action="store_true",
                            help="Whether to re-run keypoint inference even if keypoints already exist")
    parser.add_argument("--rerun-depth", action="store_true",
                            help="Whether to re-run depth inference even if depth already exist")
    parser.add_argument("--extract-reference", action="store_true",
                            help="Whether to extract features for the reference sequence (only needed if not using pre-generated features)")

    # Streaming options
    parser.add_argument("--live-davis", action="store_true",
                            help="Whether to use live DAVIS streaming")
    parser.add_argument("--target-hz", type=float, default=20.0,
                            help="Target frame rate for streaming")

    # Demo option
    parser.add_argument("--demo", action="store_true",
                            help="Whether to run a quick demo with a small subset of the data")

    args = parser.parse_args()

    # Initialize and run Event-GeM inference
    eventgem = EventGeM(args)
    eventgem.feature_inference()
    if not args.stream:
        # Generate keypoints
        eventgem.keypoint_inference()
        # Generate depth maps
        eventgem.depth_inference(args)
        
        # GT file from eventlab
        gt_file = f"{args.data_root}/{args.dataset}/ground_truth/{args.reference}_{args.query}_GT.npy"
        gt = np.load(gt_file)

        # Run re-ranking
        original, reranked, reranked_depth = eventgem.rerank_inference(args)

        # Run recall evaluation
        recall(original, reranked, reranked_depth, gt)

if __name__ == "__main__":
    main()