import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visual_odometry import odometry
from visual_odometry.io import DATA_DIR, load_image
from visual_odometry.visualization import show_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the existing visual odometry pipeline on image pairs."
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Optional explicit image paths. Provide two paths for one pair, or more paths for consecutive pairs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory of images to use when no explicit image paths are provided.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of consecutive pairs to test.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show keypoint visualizations for each tested image pair.",
    )
    return parser.parse_args()


def image_pairs(paths):
    if len(paths) < 2:
        raise ValueError("Need at least two images to form a test pair.")
    return list(zip(paths[:-1], paths[1:]))


def discover_images(data_dir):
    image_exts = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    paths = sorted(path for path in data_dir.iterdir() if path.suffix.lower() in image_exts)
    if len(paths) < 2:
        raise ValueError(f"Need at least two images in {data_dir}.")
    return paths


def run_pair(img1_path, img2_path, visualize=False):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    if visualize:
        odometry.show_images = show_images

    kp1, ds1, kp2, ds2 = odometry.calculate_keypoints(img1, img2, draw=visualize)
    if ds1 is None or ds2 is None:
        raise ValueError(f"Could not compute descriptors for {img1_path} and {img2_path}.")

    xl, xr = odometry.match_keypoints(ds1, ds2, img1, img2, kp1, kp2)
    print(
        f"PASS {img1_path.name} -> {img2_path.name}: "
        f"{len(kp1)} keypoints, {len(kp2)} keypoints"
    )
    odometry.form_epipolar_constraint(xr, xl)

def main():
    args = parse_args()
    paths = args.images if args.images else discover_images(args.data_dir)
    pairs = image_pairs(paths)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    for img1_path, img2_path in pairs:
        run_pair(img1_path, img2_path, visualize=args.visualize)

    


if __name__ == "__main__":
    main()
