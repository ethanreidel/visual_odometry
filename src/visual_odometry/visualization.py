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


def show_image_stream(images, interval=0.1):
    """Display an iterable of KITTI-style image frames in one window.

    Args:
        images: An iterable of grayscale or RGB image arrays.
        interval: Time in seconds to display each frame.
    """
    if interval <= 0:
        raise ValueError("interval must be greater than zero.")

    frames = iter(images)
    try:
        first_frame = next(frames)
    except StopIteration as exc:
        raise ValueError("show_image_stream requires at least one image.") from exc

    fig, ax = plt.subplots(figsize=(10, 4))
    image_artist = ax.imshow(first_frame, cmap="gray")
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(interval)

    for frame in frames:
        if not plt.fignum_exists(fig.number):
            break
        image_artist.set_data(frame)
        fig.canvas.draw_idle()
        plt.pause(interval)

    plt.show()
