import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import BidirectionalTransformer


class ConvBlock(nn.Module):

    def __init__(self,in_c,out_c):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(in_c,out_c,3,padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_c,out_c,3,padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self,x):

        return self.net(x)



class USeqPriorV2(nn.Module):

    def __init__(self,embed_dim=256,heads=8,layers=4):

        super().__init__()

        self.enc1 = ConvBlock(2,64)

        self.down1 = nn.Conv2d(64,128,4,2,1)

        self.enc2 = ConvBlock(128,256)

        self.down2 = nn.Conv2d(256,embed_dim,4,2,1)

        self.transformer = BidirectionalTransformer(
            embed_dim,heads,layers
        )

        self.up1 = nn.ConvTranspose2d(embed_dim,256,4,2,1)

        self.dec1 = ConvBlock(256,128)

        self.up2 = nn.ConvTranspose2d(128,64,4,2,1)

        self.dec2 = ConvBlock(64,64)

        self.out = nn.Conv2d(64,1,3,1,1)


    def forward(self,z,mask):

        x = torch.cat([z,mask],1)

        e1 = self.enc1(x)

        d1 = self.down1(e1)

        e2 = self.enc2(d1)

        d2 = self.down2(e2)

        b,c,h,w = d2.shape

        seq = d2.flatten(2).transpose(1,2)

        seq = self.transformer(seq)

        x = seq.transpose(1,2).view(b,c,h,w)

        x = self.up1(x)

        x = self.dec1(x)

        x = self.up2(x)

        x = self.dec2(x)

        return torch.sigmoid(self.out(x))