import torch
import torch.nn as nn

from torch.nn import functional as F
from transformer import Block


# GPT Model
class GPTModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed, n_head, n_layer, dropout, device):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.positional_embeddings = nn.Embedding(block_size, n_embed)

        self.decoder_blocks = nn.Sequential(*[Block(n_embed, block_size, n_head=n_head, dropout=dropout) for _ in range(n_layer)])

        self.final_layer = nn.Linear(n_embed, vocab_size)
        self.final_layernorm = nn.LayerNorm(n_embed)

        self.apply(self.__init_weights)
        self.device = device

    def __init_weights(self, module):
        """
        Initialize proper (gaussian distribution) weights for stable training and convergence
        Docs: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_
        """
        if isinstance(module, nn.Linear):
            # Initializes weights with a normal (Gaussian) distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                # Set the biases to zero
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Initializes embeddings weights with a normal (Gaussian) distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, index, targets = None):
        B, T = index.shape
        # Index and targets are both (B, T) tokens of integers
        token_embed = self.token_embeddings(index)

        # torch.arange(T) -> list of indices
        pos_embed = self.positional_embeddings(torch.arange(T, device="cuda"))   # (T, C)
        x = token_embed + pos_embed     # (B, T, C)
        x = self.decoder_blocks(x)      # (B, T, C)
        x = self.final_layernorm(x)     # (B, T, C)
        logits = self.final_layer(x)    # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Unpack logits shape to batch, seq_len, class
            B, T, C = logits.shape
            # Reshape 3D logits -> 2D logits
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # Compute loss fn
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)

            # Take only last time step
            logits = logits[:, -1, :]   # (B, C)

            # Apply softmax to get probs
            probs = F.softmax(logits, dim=-1)

            # Sample from distribution
            index_next = torch.multinomial(probs, num_samples=1)     # (B, 1)

            # Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)   # (B, T+1)

        return index
    

if __name__ == "__main__":    
    vocab_size = 3000
    block_size = 128
    n_embed = 368
    n_head = 3
    n_layer = 4
    dropout = 0.2

    model = GPTModel(vocab_size, block_size, n_embed, n_head, n_layer, dropout, device="cpu")
    dummy_input = torch.randint(0, vocab_size, (1, block_size))
    output, _ = model(dummy_input)
    print(output)