import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import FeedForward


class Block(nn.Module):
    """ Transformer Blocks """
    def __init__(self, n_embed, block_size, n_head, dropout):
        super().__init__()
        # head_size = n_embed // n_head   # Head_size to capture features
        # --------------------
        head_size = n_embed
        x_size = torch.tensor([batch_size, block_size, n_head, n_embd])
        # ------------------
        self.self_attn = MultiHeadAttention(n_embed, block_size, n_head, head_size, dropout)
        self.feed_forward = FeedForward(n_embed)
        # self.lnorm1 = nn.LayerNorm(n_embed)
        # self.lnorm2 = nn.LayerNorm(n_embed)
        self.lnorm1 = nn.LayerNorm(n_embed, head_size, dtype=torch.float16)
        self.lnorm2 = nn.LayerNorm(n_embed, head_size, dtype=torch.float16)
  

    def forward(self, x):
        y = self.self_attn(x)
        x = self.lnorm1(x+y)
        y = self.feed_forward(x)
        x = self.lnorm2(x+y)
        return x
