from pathlib import Path
import cv2

DATA_DIR = Path("/home/ethanreidel/visual_odometry/data/kitti_sample")

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
