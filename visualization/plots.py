import matplotlib.pyplot as plt
import numpy as np


def show_results(original, corrupted, mask, restored, loss):

    original = np.array(original)
    corrupted = np.array(corrupted)
    mask = np.array(mask)
    restored = np.array(restored)

    fig, ax = plt.subplots(1,5,figsize=(18,4))

    ax[0].imshow(original, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(corrupted, cmap="gray")
    ax[1].set_title("Corrupted")
    ax[1].axis("off")

    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("Mask")
    ax[2].axis("off")

    ax[3].imshow(restored, cmap="gray")
    ax[3].set_title("Restored")
    ax[3].axis("off")

    ax[4].plot(loss)
    ax[4].set_title("Loss curve")

    plt.tight_layout()

    # affichage forcé
    plt.show()

    # sauvegarde automatique
    fig.savefig("results.png")

    print("\n✔ Results saved as results.png")