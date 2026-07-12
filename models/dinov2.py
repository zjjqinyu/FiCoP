from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import DictConfig

class DINOv2(nn.Module):
    def __init__(self, cpgp_feat_size: Tuple[int, int], device : str):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2:qasfb-patch-3', 'dinov2_vitl14').to(device)
        self.device = device
        self.model.eval()
        self.feat_size = (cpgp_feat_size[0]*14, cpgp_feat_size[1]*14)

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = F.interpolate(x, size=self.feat_size, mode='bilinear', align_corners=True)
        x = self.model.forward_features(x)['x_norm_patchtokens'] # (1, 576, 1024)
        H = int(x.shape[1] ** 0.5)
        x = x.view(x.shape[0], H, H, -1).permute(0, 3, 1, 2).contiguous() # (1, 576, 16, 16)
        return x
    
    def extract_intermediate_features(self, x):
        x = F.interpolate(x, size=self.feat_size, mode='bilinear', align_corners=True)
        x = self.model.prepare_tokens_with_masks(x, None)
        H = int(x.shape[1] ** 0.5)
        features_lst = []
        for blk in self.model.blocks:
            x = blk(x)
            features_lst.append(x[:, self.model.num_register_tokens + 1 :].view(x.shape[0], H, H, -1).permute(0, 3, 1, 2).contiguous())

        x_norm = self.model.norm(x)
        final_features = x_norm[:, self.model.num_register_tokens + 1 :].view(x.shape[0], H, H, -1).permute(0, 3, 1, 2).contiguous()
        return features_lst, final_features

def get_dinov2(args: DictConfig, device: str) -> DINOv2:
    return DINOv2(cpgp_feat_size=args.cpgp.feat_size , device=device)
