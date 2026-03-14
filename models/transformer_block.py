import torch
import torch.nn as nn

class BidirectionalTransformer(nn.Module):

    def __init__(self, dim, heads, layers):

        super().__init__()

        encoder = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder,
            num_layers=layers
        )

    def forward(self, seq):

        rev = torch.flip(seq,[1])

        combined = torch.cat([seq,rev],0)

        out = self.transformer(combined)

        fwd,rev = torch.chunk(out,2,0)

        rev = torch.flip(rev,[1])

        return (fwd + rev)/2