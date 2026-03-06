import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamutils.stream as stream
from pathlib import Path
import sys
import os
THIS_DIR = Path(__file__).resolve().parent
BACKBONE_ROOT = THIS_DIR / "external" / "backbone"
sys.path.insert(0, str(BACKBONE_ROOT))

class ViTGeMExport(nn.Module):
    def __init__(self, backbone, p: float = 5.0, eps: float = 1e-6):
        super().__init__()
        self.backbone = backbone
        self.p = p
        self.eps = eps

    def forward(self, x_bchw: torch.Tensor):
        bb = self.backbone

        t = bb.patch_embed(x_bchw)
        t = t + bb.pos_embed
        t = torch.cat((bb.tokens.expand(t.shape[0], -1, -1), t), dim=1)

        for blk in bb.blocks:
            t = blk(t)
        t = bb.norm(t)

        patch_tokens = t[:, 2:, :]  # [B,N,C]
        B, N, C = patch_tokens.shape
        g = int(round(math.sqrt(N)))

        # [B,C,g,g] for GeM
        pt = patch_tokens.transpose(1, 2).reshape(B, C, g, g)
        gem = F.avg_pool2d(pt.clamp(min=self.eps).pow(self.p), (g, g)).pow(1.0 / self.p)
        out = gem.squeeze(-1).squeeze(-1)  # [B,C]

        return out

# For the ViT Backbone
def export_vit_gem_onnx(model, save_path="vitgem.onnx"):
    
    wrapper = ViTGeMExport(model)
    dummy_input = torch.randn(1, 2, 224, 224).cuda()
    torch.onnx.export(wrapper.eval(), dummy_input, save_path, 
                    input_names=['input'], output_names=['output'],
                    opset_version=18, do_constant_folding=True, dynamo=False)

# For SuperEvent (Assuming MCTS 10 channels and 256x256 crop)
def export_se_onnx(model, save_path="superevent.onnx"):
    dummy_input = torch.randn(1, 10, 240, 320).cuda()
    torch.onnx.export(model, dummy_input, save_path,
                    input_names=['mcts'], output_names=['prob', 'desc'],
                    opset_version=18, dynamo=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vit = stream.load_vit_backbone('./eventgem/ckpt/pr.pt', device)

# export to onnx
export_vit_gem_onnx(vit.eval())

# Superevent root — where the "models" package lives
SUPEREVENT_ROOT = os.path.join(THIS_DIR, "external", "superevent")

for path in (BACKBONE_ROOT, SUPEREVENT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
se_model, se_cfg = stream.build_superevent_model(Path("eventgem/external/superevent/config/super_event.yaml"), Path("eventgem/external/superevent/saved_models/super_event_weights.pth"), device)
export_se_onnx(se_model)