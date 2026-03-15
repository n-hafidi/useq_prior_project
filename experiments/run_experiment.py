import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append("/content/useq_prior_project")
sys.path.append(project_root)

import argparse
import torch
import matplotlib.pyplot as plt

from skimage import data, color, transform

from models.useq_prior_v2 import USeqPriorV2
from training.trainer import Trainer
from data.corruptions import random_mask, hole_mask, text_mask

import matplotlib.pyplot as plt

from utils.metrics import compute_psnr, compute_ssim
from utils.save_results import save_results


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
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
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

model = torch.compile(model)


cfg = {
    "training": {
        "iterations": args.iters
    },
    "loss": {
        "lambda_tv": args.lambda_tv
    }
}


trainer = Trainer(model, args.lr, cfg)


# =====================================================
# TRAIN
# =====================================================

z = torch.randn_like(clean)

loss_curve = []
psnr_curve = []

for i in range(args.iters):

    pred, loss = trainer.train(z, corrupted, mask, i)

    loss_curve.append(loss)

    # PSNR during training
    with torch.no_grad():

        current_psnr = compute_psnr(
            clean.detach().cpu().numpy().squeeze(),
            pred.detach().cpu().numpy().squeeze()
        )

    psnr_curve.append(current_psnr)

    if i % 50 == 0:
        print(f"Iter {i}/{args.iters} Loss: {loss:.6f} PSNR: {current_psnr:.2f}")


# =====================================================
# EXPERIMENT FOLDER
# =====================================================

if args.corruption == "text":
    exp_name = f"{args.image}_text_lines{args.num_lines}_th{args.thickness}_{args.size}_{args.iters}"

elif args.corruption == "random":
    exp_name = f"{args.image}_random_rate{args.missing_rate}_{args.size}_{args.iters}"

elif args.corruption == "hole":
    exp_name = f"{args.image}_hole_size{args.hole_size}_{args.size}_{args.iters}"

else:
    exp_name = f"{args.image}_unknown"

results_folder = os.path.join("results", exp_name)

os.makedirs(results_folder, exist_ok=True)

print("Results folder:", results_folder)


# =====================================================
# RESULTS
# =====================================================

restored = pred.detach().cpu().squeeze()
clean = clean.cpu().squeeze()
corrupted = corrupted.cpu().squeeze()
mask = mask.cpu().squeeze()

# =====================================================
# METRICS
# =====================================================

psnr = compute_psnr(clean.numpy(), restored.numpy())
ssim = compute_ssim(clean.numpy(), restored.numpy())

print(f"\nPSNR: {psnr:.3f}")
print(f"SSIM: {ssim:.3f}")


# =====================================================
# SAVE RESULTS
# =====================================================

save_results(
    results_folder,
    clean.numpy(),
    corrupted.numpy(),
    mask.numpy(),
    restored.numpy(),
    loss_curve,
    psnr_curve,
    psnr,
    ssim
)

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

#plt.subplot(1,5,5)
#plt.plot(loss_curve)
#plt.title("Loss")

plt.subplot(1,5,5)
plt.plot(psnr_curve)
plt.title("PSNR")


plt.tight_layout()

#plt.savefig("results.png")
plt.savefig(os.path.join(results_folder, "visualization.png"))

plt.show()


print("✔ Results saved in experiment folder")

