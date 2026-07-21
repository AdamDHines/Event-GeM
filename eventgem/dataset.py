import os
import glob
import math
import torch

import numpy as np
import eventcv as ecv
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
    
class EventGeMMCTS(Dataset):
    def __init__(self, datapath, config, offset=0):

        # create eventcv object
        self.stream = ecv.open(datapath, repr="mcts", dt_ms=config.get("dt_ms", 50), offset=offset, hot_pixel_filter=True)

        # Use same logic as load_mcts_npz on the first file
        arr0 = self.stream.slice(0).numpy()
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

    def __len__(self):
        return self.stream.n_slices

    def get_topk(self):
        return self.top_k
    
    def get_offsets(self):
        return self.off_top, self.off_left, self.off_bottom, self.off_right

    def __getitem__(self, idx):
        arr = self.stream.slice(idx).numpy()
        tensor = torch.from_numpy(arr)  # (C, H, W)

        if self.off_top == self.off_left == self.off_bottom == self.off_right == 0:
            ts_crop = tensor
        else:
            ts_crop = tensor[:, self.off_top:self.h_end, self.off_left:self.w_end]

        return ts_crop  # (C, Hc, Wc)