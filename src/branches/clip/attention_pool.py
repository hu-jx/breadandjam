import torch.nn as nn
import torch
class AttentionPool(nn.Module):
    """Learns which patch tokens matter, instead of averaging or using only CLS."""
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, patch_tokens):
        patch_tokens = self.norm(patch_tokens)
        B = patch_tokens.shape[0]
        query = self.query.expand(B, -1, -1)
        pooled, attn_weights = self.attn(query, patch_tokens, patch_tokens)
        self.last_attn_weights = attn_weights.detach()
        pooled = self.dropout(pooled.squeeze(1))
        return pooled