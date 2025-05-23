import torch
import torch.nn as nn

from torch.nn import functional as F
from .flash_attention import MultiHeadFlashAttn

class Head(nn.Module):
    """ Head for Self-Attention (Scaled-Dot Product) """
    def __init__(self, n_embed, block_size, head_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # NOTE: Register a lower triangular matrix as a buffer (used for masking future tokens in self-attention)
        # It’s non-trainable, included in state_dict(), and avoids recomputation.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input: (B, T, C) -> Output: (B, T, head_size)
        B, T, C = x.shape                   # Unpack Input Dimensions
        k, q = self.key(x), self.query(x)   # (B, T, head_size)

        # Compute attn_scores(attn_weights) [creating q @ transposed k grid matrix]
        w = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5             # (B, T, head_size) @ # (B, head_size, T) -> (B, T, T)

        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # (B, T, T)
        w = F.softmax(w, dim=-1)                                    # (B, T, T)
        w = self.dropout(w)

        # Perform the weighted aggregation of the values
        v = self.value(x)                   # (B, T, head_size)
        out = w @ v                         # (B, T, T) @ # (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of Attn in Parallel """
    def __init__(self, n_embed, block_size, num_heads, head_size, dropout=0.2):
        super().__init__()
        # self.heads = nn.ModuleList([Head(n_embed, block_size, head_size, dropout) for _ in range(num_heads)])  # Create heads in parallel
        # -----------
        self.mha_flash_attn = MultiHeadFlashAttn(head_size)
        # -----------
        self.proj = nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B, T, C) -> Concat feature(last_dim): (B, T, [h0_1, h0_2, h0_3, h0_4, h1_1, h1_2, h1_3, h1_4, h2_1, h2_2, h2_3, h2_4])
        # out = torch.cat([h(x) for h in self.heads], dim=-1)
        # -----------
        out = self.mha_flash_attn(x)
        # -------------
        out = self.dropout(self.proj(out))
        return out