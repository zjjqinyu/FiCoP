from functools import partial
from typing import Tuple, Callable
from collections import OrderedDict

from sympy import Q
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import Encoder, MLPBlock
from omegaconf import DictConfig

class CrossAttentionBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        torch._assert(q.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {q.shape}")
        torch._assert(kv.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {kv.shape}")
        x = self.ln_1(q)
        x2 = self.ln_1(kv)
        x, _ = self.self_attention(query=x, key=x2, value=x2, need_weights=False)
        x = self.dropout(x)
        x = x + q

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class CrossAttention(Encoder):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, norm_layer)
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = CrossAttentionBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    
class PatchCorrsHead(nn.Module):
    def __init__(self, in_feat_size: int, grid_size: int, in_channels: int, conv_channels: int, num_layers: int, num_heads: int, mlp_dim: int, device:str):
        super().__init__()
        self.in_feat_size = in_feat_size
        self.grid_size = grid_size
        assert in_feat_size % grid_size == 0, f"in_feat_size {in_feat_size} should be divisible by grid_size {grid_size}"
        self.patch_size = in_feat_size//grid_size

        self.conv1 = conv_block(self.patch_size**2, conv_channels)
        self.conv2 = conv_block(conv_channels, conv_channels//2)
        self.conv3 = conv_block(conv_channels//2, conv_channels//4)
        self.conv4 = conv_block(conv_channels//4, conv_channels//8)
        self.conv5 = nn.Conv2d(conv_channels//8, 1, kernel_size=self.patch_size, stride=self.patch_size)
        self.device = device

    def forward(self, feat_a, feat_q):
        x = torch.einsum('bchw, bcyz -> bhwyz', feat_a, feat_q) # (B, H, W, H, W)
        x = self.split_into_patches(x) # (B, grid_size, grid_size, patch_size**2, H, W)
        B,G1,G2,P,H,W = x.shape # P=patch_size**2
        x = x.reshape(B*G1*G2, P, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)  # (B*grid_size*grid_size, 1, grid_size, grid_size)
        corrs_logits = x.reshape(B, G1*G2, G1*G2) # (B, grid_size*grid_size, grid_size*grid_size)
        patch_corrs_map = corrs_logits
        return patch_corrs_map
    
    def split_into_patches(self, x):
        B, H1, W1, H2, W2 = x.shape
        grid_size = self.grid_size
        patch_h = H1 // grid_size
        patch_w = W1 // grid_size
        
        x = x.view(B, grid_size, patch_h, grid_size, patch_w, H2, W2)
        x = x.permute(0, 1, 3, 2, 4, 5, 6)
        x = x.reshape(B, grid_size, grid_size, patch_h * patch_w, H2, W2)
        
        return x
    
def get_patch_corrs_head(args: DictConfig, device:str) ->PatchCorrsHead:
    return PatchCorrsHead(args.cpgp.feat_size[0], 
                          args.patch_corrs_head.grid_size, 
                          args.cpgp.in_channels, 
                          args.cpgp.conv_channels,
                          args.patch_corrs_head.num_layers, 
                          args.patch_corrs_head.num_heads, 
                          args.patch_corrs_head.mlp_dim, 
                          device)