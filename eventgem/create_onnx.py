import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamutils.stream as stream

class ViTGeMExport(nn.Module):
    def __init__(self, backbone, p: float = 5.0, eps: float = 1e-6):
        super().__init__()
        self.backbone = backbone
        self.p = p
        self.eps = eps

    def forward(self, x_bchw: torch.Tensor, return_heatmap: bool = True):
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

        if not return_heatmap:
            return out

        # Heatmap: per-token L2 norm -> [B,g,g]
        hm = patch_tokens.norm(dim=-1).reshape(B, g, g)  # float
        # normalize 0..1 per image
        hm = hm - hm.amin(dim=(1, 2), keepdim=True)
        hm = hm / (hm.amax(dim=(1, 2), keepdim=True) + 1e-6)
        return out, hm

# For the ViT Backbone
def export_vit_gem_onnx(model, save_path="vitgem-hm.onnx"):
    
    wrapper = ViTGeMExport(model)
    dummy_input = torch.randn(1, 2, 224, 224).cuda().half()
    torch.onnx.export(wrapper, dummy_input, save_path, 
                    input_names=['input'], output_names=['output'],
                    opset_version=18, do_constant_folding=True)

# For SuperEvent (Assuming MCTS 10 channels and 256x256 crop)
def export_se_onnx(model, save_path="superevent.onnx"):
    dummy_input = torch.randn(1, 10, 256, 256).cuda().half()
    torch.onnx.export(model, dummy_input, save_path,
                    input_names=['mcts'], output_names=['prob', 'desc'],
                    opset_version=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vit = stream.load_vit_backbone('./eventgem/ckpt/pr.pt', device)
vitH, vitW = stream.infer_vit_input_hw(vit)

# export to onnx
export_vit_gem_onnx(vit.half())

# se_model, se_cfg = stream.build_superevent_model(Path("eventgem/external/superevent/config/super_event.yaml"), Path("eventgem/external/superevent/saved_models/super_event_weights.pth"), device)
# export_se_onnx(se_model.half())