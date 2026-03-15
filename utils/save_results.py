import os
import matplotlib.pyplot as plt


def save_results(folder,
                 clean,
                 corrupted,
                 mask,
                 restored,
                 loss_curve,
                 psnr_curve,
                 psnr,
                 ssim):

    os.makedirs(folder, exist_ok=True)

    # ==============================
    # SAVE IMAGES
    # ==============================

    plt.imsave(f"{folder}/original.png", clean, cmap="gray")
    plt.imsave(f"{folder}/corrupted.png", corrupted, cmap="gray")
    plt.imsave(f"{folder}/mask.png", mask, cmap="gray")
    plt.imsave(f"{folder}/restored.png", restored, cmap="gray")

    # ==============================
    # LOSS CURVE
    # ==============================

    plt.figure()
    plt.plot(loss_curve)
    plt.title("Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"{folder}/loss_curve.png")
    plt.close()

    # ==============================
    # PSNR CURVE
    # ==============================

    plt.figure()
    plt.plot(psnr_curve)
    plt.title("PSNR Curve")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.savefig(f"{folder}/psnr_curve.png")
    plt.close()

    # ==============================
    # METRICS
    # ==============================

    with open(f"{folder}/metrics.txt", "w") as f:
        f.write(f"PSNR: {psnr}\n")
        f.write(f"SSIM: {ssim}\n")

    print(f"✔ Results saved in {folder}")