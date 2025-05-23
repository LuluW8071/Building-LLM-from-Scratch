import torch.nn as nn


class FeedForward(nn.Module):
    """ Linear Layers follwed by non-linearity """
    def __init__(self, n_embed, dropout=0.2):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.linear_layers(x)