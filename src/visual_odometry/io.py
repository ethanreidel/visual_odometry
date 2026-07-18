from pathlib import Path
import numpy as np
import cv2

SEQUENCE_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "kitti_odometry_sample" / "sequences" / "00"
)
DATA_DIR = SEQUENCE_DIR / "image_0"
CALIB_PATH = SEQUENCE_DIR / "calib.txt"

def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def load_projection_matrix(path=CALIB_PATH, camera="P0"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not read calibration data: {path}")

    for line in path.read_text().splitlines():
        key, values = line.split(":", 1)
        if key == camera:
            return np.array(values.split(), dtype=float).reshape(3, 4)

    raise ValueError(f"Camera {camera!r} not found in calibration file: {path}")


def load_intrinsic_matrix(path=CALIB_PATH, camera="P0"):
    projection = load_projection_matrix(path, camera)
    return projection[:, :3]
