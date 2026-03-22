"""
Vision Transformer (ViT) for Visual Servoing
Divides input images into patches, projects them into an embedding space,
and processes them through a standard Transformer encoder.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly projects each patch.

    Args:
        image_size  (int): Height/width of input image (assumed square)
        patch_size  (int): Height/width of each patch (assumed square)
        in_channels (int): Number of image channels
        embed_dim   (int): Output embedding dimension per patch
    """
    def __init__(self, image_size: int = 8, patch_size: int = 2,
                 in_channels: int = 3, embed_dim: int = 256):
        super(PatchEmbedding, self).__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        x = self.projection(x)          # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)                # (B, embed_dim, num_patches)
        return x.transpose(1, 2)        # (B, num_patches, embed_dim)


class ViT(nn.Module):
    """
    Vision Transformer adapted for visual servoing weight prediction.

    Architecture:
        PatchEmbedding → [CLS token + Positional Encoding]
        → TransformerEncoder (depth layers, nhead attention heads)
        → MLP head on CLS token → output weights

    Args:
        image_size  (int): Input image size (default: 8)
        patch_size  (int): Patch size (default: 2)
        in_channels (int): Image channels (default: 3)
        embed_dim   (int): Transformer embedding dim (default: 256)
        depth       (int): Number of transformer layers (default: 6)
        nhead       (int): Number of attention heads (default: 8)
        mlp_ratio   (float): MLP hidden dim ratio (default: 4.0)
        output_size (int): Number of output weights (default: 70)
    """
    def __init__(
        self,
        image_size: int = 8,
        patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        nhead: int = 8,
        mlp_ratio: float = 4.0,
        output_size: int = 70,
    ):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            norm_first=True,       # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                                         # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                                  # (B, N+1, D)
        x = x + self.pos_embed                                           # add positional encoding
        x = self.transformer(x)                                          # (B, N+1, D)
        x = self.norm(x[:, 0])                                           # CLS token only
        return self.head(x)                                              # (B, output_size)
