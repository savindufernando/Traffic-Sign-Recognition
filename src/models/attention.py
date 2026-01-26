"""
Attention Mechanisms for Traffic Sign Recognition.
Includes SE-Block and CBAM attention modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    
    Adaptively recalibrates channel-wise feature responses by explicitly 
    modeling interdependencies between channels.
    
    Paper: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Combines channel and spatial attention for enhanced feature learning.
    
    Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(
        self, 
        channels: int, 
        reduction: int = 16, 
        kernel_size: int = 7
    ):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention
        x = x * self.spatial_attention(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Custom Transformer Encoder Layer with pre-norm architecture.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder with multiple layers.
    
    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        dim_feedforward: FFN hidden dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1536,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
