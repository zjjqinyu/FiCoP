from collections import OrderedDict
from typing import Callable, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vision_transformer import MLPBlock
from omegaconf import DictConfig

class EncoderBlock(nn.Module):
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

    def forward(self, input: torch.Tensor, attn_mask: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False, attn_mask=attn_mask)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

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
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        torch._assert(q.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {q.shape}")
        torch._assert(kv.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {kv.shape}")
        x = self.ln_1(q)
        x2 = self.ln_1(kv)
        x, _ = self.attention(query=x, key=x2, value=x2, need_weights=False)
        x = self.dropout(x)
        x = x + q

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

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
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
            layers[f"cross_attention_layer_{i}"] = CrossAttentionBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.ModuleDict(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, attn_mask: torch.Tensor, text_feat: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        input = self.dropout(input)
        for layer in self.layers.values():
            if isinstance(layer, EncoderBlock):
                input = layer(input, attn_mask)
            else:
                input = layer(input, text_feat)
        output =  self.ln(input)
        return output

# Cross-perspective Global Perception module
class CPGPModule(nn.Module):
    def __init__(self, feat_size: Tuple[int, int], in_channels: int, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, text_feat_dim: int, device:str):
        super().__init__()
        self.feat_size = feat_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        seq_length = 2 * feat_size[0] * feat_size[1]
        self.encoder = Encoder(seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.1, attention_dropout=0.1, norm_layer=nn.LayerNorm).to(device)
        self.down_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1).to(device)
        self.up_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, stride=1).to(device)
        self.text_feat_proj = nn.Sequential(
            nn.Linear(text_feat_dim, hidden_dim).to(device),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, feat_a, feat_q, text_feat):
        B,C,H,W = feat_a.shape
        feat_a = self.down_conv(feat_a)
        feat_q = self.down_conv(feat_q)
        feat_a = feat_a.reshape(B, self.hidden_dim, H*W).permute(0, 2, 1)
        feat_q = feat_q.reshape(B, self.hidden_dim, H*W).permute(0, 2, 1)
        input_feat = torch.cat([feat_a, feat_q], dim=1)
        text_feat = self.text_feat_proj(text_feat)
        out = self.encoder(input_feat, None, text_feat)
        out_a = out[:, :H*W, :]
        out_q = out[:, H*W:, :]
        out_a = out_a.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)
        out_q = out_q.permute(0, 2, 1).reshape(B, self.hidden_dim, H, W)
        out_a = self.up_conv(out_a)
        out_q = self.up_conv(out_q)
        return out_a, out_q

def get_cpgp_module(args: DictConfig, device:str) ->CPGPModule:
    return CPGPModule(args.cpgp.feat_size, args.cpgp.in_channels, args.cpgp.num_layers, args.cpgp.num_heads, args.cpgp.hidden_dim, args.cpgp.mlp_dim, 768, device)