import os
import argparse
import numpy as np
from eventgem.analysis import recall
from eventgem.feature_extraction import EventGeM

def main():
    parser = argparse.ArgumentParser(description="Event-GeM: Pre-trained feature extraction and 2D-homology re-reanking for visual place recognition")
    
    # Dataset parameters
    parser.add_argument("--dataset", "-d", type=str, required=True, 
                            choices=["brisbane_event", "nsavp", "qcr-event", "fast-slow"],
                            help="Dataset to use for evaluation")
    parser.add_argument("--reference", "-r", type=str, required=True, 
                            help="Reference directory to use for evaluation")
    parser.add_argument("--query", "-q", type=str, required=True, 
                            help="Query directory to use for evaluation")
    parser.add_argument("--recon-msec", type=int, default=50,
                            help="Reconstruction time window in milliseconds for event datasets")
    parser.add_argument("--data-root", type=str, default="/media/adam/vprdatasets/eventlab/brisbane_event", 
                            help="Root directory for datasets")
    
    # Model parameters
    parser.add_argument("--backbone-ckpt", type=str, default="./eventgem/ckpt/pr.pt",
                            help="Path to the backbone checkpoint")
    parser.add_argument("--top-k", type=int, default=100,
                            help="Number of top candidates to re-rank using 2D-homology")
    parser.add_argument("--ransac-thresh", type=float, default=5.0, 
                            help="RANSAC pixel threshold (e.g. 3-5 px)")
    parser.add_argument("--inlier-weight", type=float, default=0.05, 
                            help="Distance subtraction per inlier")
    parser.add_argument("--batch-size", type=int, default=32,
                            help="Batch size for feature extraction")
    parser.add_argument("--feature-out", type=str, default="./eventgem/features",
                            help="Directory to save extracted features")
    parser.add_argument("--keypoint-out", type=str, default="./eventgem/keypoints",
                            help="Directory to save detected keypoints")
    
    # Togglable operation
    parser.add_argument("--mode", type=str, default="feature-extract", choices=["feature-extract", "keypoints"],
                            help="Operation mode: feature extraction or keypoint detection")
    parser.add_argument("--re-extract", action="store_true",
                            help="Re-extract features/keypoints even if they already exist")
    args = parser.parse_args()

    # Initialize and run Event-GeM inference
    eventgem = EventGeM(args)
    if args.mode == "feature-extract":
        eventgem.feature_inference()
    else:
        original, reranked = eventgem.keypoint_inference()
        # Run Recall@K evaluation
        gt = np.load(os.path.join(args.data_root, "ground_truth", f"{args.reference}_{args.query}_GT.npy"))
        recall(original, reranked, gt)

if __name__ == "__main__":
    main()