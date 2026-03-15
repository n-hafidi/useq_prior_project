import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/content/useq_prior_project")

import argparse
import torch
import matplotlib.pyplot as plt

from skimage import data, color, transform

from models.useq_prior_v2 import USeqPriorV2
from training.trainer import Trainer
from data.corruptions import random_mask, hole_mask, text_mask

# =====================================================
# ARGUMENTS
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument("--image", type=str, default="camera",
                    choices=["camera", "astronaut", "coins", "moon"])

parser.add_argument("--size", type=int, default=256)

parser.add_argument("--corruption", type=str, default="text",
                    choices=["text", "random", "hole"])

parser.add_argument("--num_lines", type=int, default=3)
parser.add_argument("--thickness", type=int, default=10)
parser.add_argument("--missing_rate", type=float, default=0.4)
parser.add_argument("--hole_size", type=int, default=80)

parser.add_argument("--iters", type=int, default=2000)
parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--lambda_tv", type=float, default=0.1)

args = parser.parse_args()


# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =====================================================
# IMAGE
# =====================================================

images = {
    "camera": data.camera(),
    "astronaut": data.astronaut(),
    "coins": data.coins(),
    "moon": data.moon()
}

img = images[args.image]

if img.ndim == 3:
    img = color.rgb2gray(img)

img = transform.resize(img, (args.size, args.size))

clean = torch.tensor(img).float().unsqueeze(0).unsqueeze(0).to(device)


# =====================================================
# CORRUPTION
# =====================================================

if args.corruption == "hole":

    mask = hole_mask(clean, args.hole_size)

elif args.corruption == "random":

    mask = random_mask(clean, args.missing_rate)

elif args.corruption == "text":

    mask = text_mask(clean, args.num_lines, args.thickness)

else:
    raise ValueError("Unknown corruption")

corrupted = clean * mask


# =====================================================
# MODEL
# =====================================================

model = USeqPriorV2().to(device)


cfg = {
    "training": {
        "iterations": args.iters
    },
    "loss": {
        "lambda_tv": args.lambda_tv
    }
}


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
trainer = Trainer(model, optimizer, cfg)


# =====================================================
# TRAIN
# =====================================================

z = torch.randn_like(clean)

loss_curve = []

for i in range(args.iters):

    pred, loss = trainer.train(z, corrupted, mask, i)

    loss_curve.append(loss)

    if i % 50 == 0:
        print(f"Iter {i}/{args.iters} Loss: {loss:.6f}")


# =====================================================
# RESULTS
# =====================================================

restored = pred.detach().cpu().squeeze()
clean = clean.cpu().squeeze()
corrupted = corrupted.cpu().squeeze()
mask = mask.cpu().squeeze()


# =====================================================
# DISPLAY
# =====================================================

plt.figure(figsize=(18,4))

plt.subplot(1,5,1)
plt.imshow(clean, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,5,2)
plt.imshow(corrupted, cmap="gray")
plt.title("Corrupted")
plt.axis("off")

plt.subplot(1,5,3)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1,5,4)
plt.imshow(restored, cmap="gray")
plt.title("Restored")
plt.axis("off")

plt.subplot(1,5,5)
plt.plot(loss_curve)
plt.title("Loss")

plt.tight_layout()

plt.savefig("results.png")

plt.show()

print("✔ Results saved as results.png")