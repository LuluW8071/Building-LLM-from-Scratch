import torch

from flash_attn import flash_attn_func
from torch import nn

class MultiHeadFlashAttn(nn.Module):
    def __init__(self, head_size, n_embed, num_heads, block_size, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = nn.Linear(n_embed//num_heads, head_size, bias=False, dtype=torch.float16)
        self.query = nn.Linear(n_embed//num_heads, head_size, bias=False, dtype=torch.float16)
        self.value = nn.Linear(n_embed//num_heads, head_size, bias=False, dtype=torch.float16)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, num_heads):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        x = x.view(B, T, num_heads, C // num_heads)  # (B,T,C) -> (B,T,hs,C/hs)
        # print(x.shape)
        x = x.transpose(1, 2)  
        
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        # out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = flash_attn_func(
            q, k, v, 
            dropout_p=0.0, 
            softmax_scale=None, 
            causal=True,
            window_size=(-1, -1), 
            alibi_slopes=None, 
            deterministic=False
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)    # (B, T, nh * hs)
        return out