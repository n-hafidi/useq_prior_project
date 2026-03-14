import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import yaml
from skimage import data,transform
import torch.optim as optim

from models.useq_prior_v2 import USeqPriorV2
from data.corruptions import *
from training.trainer import Trainer
from visualization.plots import show_results


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


with open("configs/default.yaml") as f:
    cfg=yaml.safe_load(f)


img=data.camera()

img=transform.resize(img,(cfg["experiment"]["size"],cfg["experiment"]["size"]))

clean=torch.tensor(img).float().view(1,1,*img.shape).to(device)


if cfg["corruption"]["type"]=="hole":

    mask=hole_mask(clean,cfg["corruption"]["hole_size"])

elif cfg["corruption"]["type"]=="random":

    mask=random_mask(clean,cfg["corruption"]["missing_rate"])

else:

    mask=text_mask(
        clean,
        cfg["corruption"]["num_lines"],
        cfg["corruption"]["thickness"]
    )


mask=mask.to(device)

corrupted=clean*mask


#model=USeqPriorV2().to(device)
model = USeqPriorV2(
    embed_dim=cfg["model"]["embed_dim"],
    heads=cfg["model"]["heads"],
    layers=cfg["model"]["transformer_layers"]
).to(device)

opt=optim.Adam(model.parameters(),lr=cfg["training"]["lr"])

z=torch.randn_like(clean).to(device)

trainer=Trainer(model,opt,cfg)

pred=trainer.train(z,corrupted,mask)

restored=corrupted*mask+pred*(1-mask)


show_results(
    clean.squeeze().cpu(),
    corrupted.squeeze().cpu(),
    mask.squeeze().cpu(),
    restored.squeeze().cpu(),
    trainer.loss_curve
)