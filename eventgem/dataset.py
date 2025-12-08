import os
import glob
import math
import torch

import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset

class EventGeMData(Dataset):
    def __init__(self, datapath):
        self.path = datapath

        # Identify the files in the dataset
        self.files_sort = sorted(glob.glob(os.path.join(self.path, "**", "*.npy"), recursive=True))
        if len(self.files_sort) == 0:
            raise RuntimeError(f"No event frames found in {self.path} or its subdirectories")
        
        # Ignore files with the name "event_frame_times_ticks.npy"
        self.files_sort = [f for f in self.files_sort if not f.endswith("event_frame_times_ticks.npy")]
        
    def __len__(self):
        return len(self.files_sort)
    
    def __getitem__(self, idx):
        """Loads event frame, handles (H,W,C)->(C,H,W), robust norm."""
        data = np.load(self.files_sort[idx])
        tensor = torch.from_numpy(data).float() 
        
        # Permute if H,W,C
        if tensor.shape[-1] in [2, 3] and tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
            
        # Resize
        tensor = tensor.unsqueeze(0) # Batch dim
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)

        # Robust Norm (98th percentile)
        flat = tensor.view(-1)
        if flat.numel() > 0:
            k = int(0.98 * flat.numel())
            robust_max, _ = torch.kthvalue(flat, k)
            if robust_max < 1e-6: robust_max = 1.0
            tensor = torch.clamp(tensor, max=robust_max)
            tensor = tensor / robust_max
            tensor = tensor * 2 - 1

        return tensor
    
class EventGeMMCTS(Dataset):
    def __init__(self, datapath, config):
        self.root = Path(datapath)

        # Match list_mcts_files
        self.files_sort = sorted(self.root.glob("mcts_*.npz"))
        if not self.files_sort:
            raise RuntimeError(f"No mcts_*.npz files found in {self.root}")

        # Use same logic as load_mcts_npz on the first file
        arr0 = self._load_mcts_npz(self.files_sort[0])
        _, self.H, self.W = arr0.shape  # (C, H, W)

        # Inline build_crop_mask
        max_factor_required = config["grid_size"]
        if "backbone_config" in config:
            stage_blocks = config["backbone_config"]["num_blocks"]
            patch_size = config["backbone_config"]["stem"]["patch_size"]
            downsample_factor = patch_size * (2 ** (len(stage_blocks) - 1))
            max_factor_required = downsample_factor

            if "attention" in config["backbone_config"]["stage"]:
                max_partition = np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
                max_factor_required *= max_partition

        crop = np.array([self.H, self.W]) % max_factor_required

        self.off_top = math.ceil(crop[0] / 2)
        self.off_bottom = math.floor(crop[0] / 2)
        self.off_left = math.ceil(crop[1] / 2)
        self.off_right = math.floor(crop[1] / 2)

        self.Hc = self.H - crop[0]
        self.Wc = self.W - crop[1]

        self.h_end = self.H - self.off_bottom if self.off_bottom > 0 else self.H
        self.w_end = self.W - self.off_right if self.off_right > 0 else self.W

        self.top_k = max(self.Hc, self.Wc) // 2

    def _load_mcts_npz(self, path: Path) -> np.ndarray:
        data = np.load(str(path))
        if "mcts" in data:
            arr = data["mcts"]
        else:
            first_key = list(data.keys())[0]
            arr = data[first_key]
        return arr.astype(np.float32)  # (C, H, W) ideally

    def __len__(self):
        return len(self.files_sort)

    def get_topk(self):
        return self.top_k
    
    def get_offsets(self):
        return self.off_top, self.off_left, self.off_bottom, self.off_right

    def __getitem__(self, idx):
        arr = self._load_mcts_npz(self.files_sort[idx])
        tensor = torch.from_numpy(arr)  # (C, H, W)

        if self.off_top == self.off_left == self.off_bottom == self.off_right == 0:
            ts_crop = tensor
        else:
            ts_crop = tensor[:, self.off_top:self.h_end, self.off_left:self.w_end]

        return ts_crop  # (C, Hc, Wc)