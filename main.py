import os
import argparse
import numpy as np
from eventgem.analysis import recall
from eventgem.feature_extraction import EventGeM

def main():
    parser = argparse.ArgumentParser(description="Event-GeM: Pre-trained feature extraction and 2D-homology re-reanking for visual place recognition")
    
    # Dataset parameters
    # parser.add_argument("--dataset", "-d", type=str,  
    #                         choices=["brisbane_event", "nsavp", "fast_slow"],
    #                         help="Dataset to use for evaluation")
    # parser.add_argument("--reference", "-r", type=str, 
    #                         help="Reference directory to use for evaluation")
    # parser.add_argument("--query", "-q", type=str, 
    #                         help="Query directory to use for evaluation")
    parser.add_argument("--dataset", "-d", type=str, default="brisbane_event",
                            choices=["brisbane_event", "nsavp", "fast_slow"],
                            help="Dataset to use for evaluation")
    parser.add_argument("--reference", "-r", type=str, default="sunset2",
                            help="Reference directory to use for evaluation")
    parser.add_argument("--query", "-q", type=str, default="sunset1",
                            help="Query directory to use for evaluation")
    parser.add_argument("--recon-msec", type=int, default=50,
                            help="Reconstruction time window in milliseconds for event datasets")
    parser.add_argument("--mcts-time", type=float, nargs='+', default=[10, 20, 30, 40, 50],
                            help="Space-separated list of temporal window sizes in msec.")
    parser.add_argument("--data-root", type=str, default="/media/adam/vprdatasets/eventgem", 
                            help="Root directory for datasets")
    
    # Model parameters
    parser.add_argument("--backbone-ckpt", type=str, default="./eventgem/ckpt/pr.pt",
                            help="Path to the backbone checkpoint")
    parser.add_argument("--top-k", type=int, default=50,
                            help="Number of top candidates to re-rank using 2D-homography")
    parser.add_argument("--ransac-thresh", type=float, default=5.0, 
                            help="RANSAC pixel threshold (e.g. 3-5 px)")
    parser.add_argument("--inlier-weight", type=float, default=0.05, 
                            help="Distance subtraction per inlier")
    parser.add_argument("--backbone_batch-size", type=int, default=32,
                            help="Batch size for feature extraction")
    parser.add_argument("--keypoint_batch-size", type=int, default=16,
                            help="Batch size for feature extraction")
    parser.add_argument("--feature-out", type=str, default="./eventgem/features",
                            help="Directory to save extracted features")
    parser.add_argument("--keypoint-out", type=str, default="./eventgem/keypoints",
                            help="Directory to save detected keypoints")
    parser.add_argument("--rerank_mode", type=str, default="depth", choices=["keypoints", "depth", "both"],
                            help="Which results to evaluate: original, reranked, or both")
    parser.add_argument("--reference_depth_dir", type=str, default="/media/adam/vprdatasets/edvpr/sunset2/raw16",
                            help="Directory containing reference depth maps (if using depth-based re-ranking)")
    parser.add_argument("--query_depth_dir", type=str, default="/media/adam/vprdatasets/edvpr/sunset1/raw16",
                            help="Directory containing query depth maps (if using depth-based re-ranking)")
    parser.add_argument("--depth_pattern", type=str, default="depth_{:06d}.png",
                            help="File pattern for depth maps")
    # Togglable operation
    parser.add_argument("--mode", type=str, default="keypoints", choices=["feature-extract", "keypoints"],
                            help="Operation mode: feature extraction or keypoint detection")
    args = parser.parse_args()

    # Initialize and run Event-GeM inference
    eventgem = EventGeM(args)
    if args.mode == "feature-extract":
        eventgem.feature_inference()
    else:
        original, reranked = eventgem.keypoint_inference()
        # Run Recall@K evaluation
        gt = np.load(os.path.join(args.data_root, args.dataset, "ground_truth", f"{args.reference}_{args.query}_GT.npy"))
        # gt = None
        recall(original, reranked, gt)

if __name__ == "__main__":
    main()