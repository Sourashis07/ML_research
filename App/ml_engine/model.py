import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1dTemporalEncoder(nn.Module):
    """
    1D-CNN temporal feature extractor for local transient spikes and feature extraction.
    Input shape: [B, W, C] -> reshaped to [B, C, W] for Conv1d
    Output shape: [B, hidden_dim, W]
    """
    def __init__(self, num_channels: int, hidden_dim: int = 64, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, C] -> permute to [B, C, W]
        x_perm = x.permute(0, 2, 1)
        feat = self.act1(self.bn1(self.conv1(x_perm)))
        return feat  # [B, hidden_dim, W]


class BiGRUContextEncoder(nn.Module):
    """
    Bidirectional GRU for modeling long-term sequence dynamics in both forward and backward directions.
    Input: [B, W, hidden_dim]
    Output: [B, W, hidden_dim]
    """
    def __init__(self, hidden_dim: int = 64, gru_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, hidden_dim]
        out, _ = self.gru(x)  # [B, W, hidden_dim]
        return out


class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-Head Cross-Channel Self-Attention for spatial dependency modeling across sensors.
    Input: [B, W, hidden_dim]
    Output: [B, W, hidden_dim], attention_weights [B, num_heads, W, W]
    """
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor):
        # x: [B, W, hidden_dim]
        attn_out, attn_weights = self.mha(x, x, x)
        x_out = self.norm(x + attn_out)
        return x_out, attn_weights


class SpatialTemporalAutoencoder(nn.Module):
    """
    Track A: Spatial-Temporal Attention Autoencoder
    Architecture:
    1. 1D-CNN Temporal Feature Extraction
    2. Bidirectional GRU Temporal Modeling
    3. Multi-Head Cross-Channel Self-Attention
    4. Symmetric Decoder network reconstructing back to [B, W, C]
    """
    def __init__(self, num_channels: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # Encoder
        self.cnn_encoder = Conv1dTemporalEncoder(num_channels=num_channels, hidden_dim=hidden_dim)
        self.bigru_encoder = BiGRUContextEncoder(hidden_dim=hidden_dim)
        self.attention = MultiHeadSpatialAttention(hidden_dim=hidden_dim, num_heads=num_heads)

        # Decoder
        self.decoder_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_channels)
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Input x: [B, W, C]
        Returns:
            x_recon: [B, W, C] (reconstructed telemetry sequence)
            attn_weights: Cross-attention matrix weights
        """
        B, W, C = x.shape

        # 1. 1D-CNN temporal feature extraction -> [B, hidden_dim, W]
        cnn_feat = self.cnn_encoder(x)
        cnn_feat_seq = cnn_feat.permute(0, 2, 1)  # [B, W, hidden_dim]

        # 2. BiGRU contextual modeling -> [B, W, hidden_dim]
        gru_feat = self.bigru_encoder(cnn_feat_seq)

        # 3. Multi-Head Spatial Self-Attention -> [B, W, hidden_dim]
        attn_feat, attn_weights = self.attention(gru_feat)

        # 4. Decoder Reconstruction -> [B, W, C]
        dec_out, _ = self.decoder_gru(attn_feat)
        x_recon = self.decoder_head(dec_out)

        return x_recon, attn_weights
