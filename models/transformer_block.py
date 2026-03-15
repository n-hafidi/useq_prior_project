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

        return self.transformer(seq)