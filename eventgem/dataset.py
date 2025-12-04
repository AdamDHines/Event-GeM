import os
import glob
import torch

import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset

class EventGeMData(Dataset):
    def __init__(self, datapath):
        self.path = datapath

        # Identify the files in the dataset
        self.files_sort = sorted(glob.glob(os.path.join(self.path, "**", "*.npy"), recursive=True))
        if len(self.files_sort) == 0:
            raise RuntimeError(f"No event frames found in {self.path} or its subdirectories")
        
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