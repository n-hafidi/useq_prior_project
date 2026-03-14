import matplotlib.pyplot as plt


def show_results(original,corrupted,mask,restored,loss):

    fig,ax=plt.subplots(1,5,figsize=(18,4))

    ax[0].imshow(original,cmap="gray")
    ax[0].set_title("Original")

    ax[1].imshow(corrupted,cmap="gray")
    ax[1].set_title("Corrupted")

    ax[2].imshow(mask,cmap="gray")
    ax[2].set_title("Mask")

    ax[3].imshow(restored,cmap="gray")
    ax[3].set_title("Restored")

    ax[4].plot(loss)
    ax[4].set_title("Loss")

    plt.show()