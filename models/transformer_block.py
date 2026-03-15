import torch
import torch.nn as nn


class BidirectionalTransformer(nn.Module):

    def __init__(self, dim, heads, layers):
        super().__init__()

        encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder,
            num_layers=layers
        )

    def forward(self, seq):

        forward = self.transformer(seq)

        reverse = torch.flip(seq, [1])
        reverse = self.transformer(reverse)
        reverse = torch.flip(reverse, [1])

        return 0.5 * (forward + reverse)