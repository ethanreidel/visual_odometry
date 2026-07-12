import matplotlib.pyplot as plt

def show_images(*images):
    if len(images) == 0:
        raise ValueError("show_image requires at least one image.")

    fig, axes = plt.subplots(len(images), 1, figsize=(10, 4 * len(images)))
    if len(images) == 1:
        axes = [axes]

    for ax, img in zip(axes, images):
        ax.imshow(img, cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
