import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visual_odometry import odometry
from visual_odometry.io import load_image
from visual_odometry.visualization import show_image_stream


DEMO_DATA_DIR = REPO_ROOT / "data" / "kitti_sample"


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


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
        default=DEMO_DATA_DIR,
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
        help="Overlay SIFT + FLANN motion tracks on each current frame.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between match visualizations (default: 0.1).",
    )
    parser.add_argument(
        "--max-display-matches",
        type=positive_int,
        default=None,
        metavar="N",
        help="Show at most N strongest matches without changing odometry (default: all).",
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


def run_pair(img1_path, img2_path, visualize=False, max_display_matches=None):
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    kp1, ds1, kp2, ds2 = odometry.calculate_keypoints(img1, img2)
    if ds1 is None or ds2 is None:
        raise ValueError(f"Could not compute descriptors for {img1_path} and {img2_path}.")

    match_result = odometry.match_keypoints(
        ds1,
        ds2,
        img1,
        img2,
        kp1,
        kp2,
        return_visualization=visualize,
        max_display_matches=max_display_matches,
    )
    if visualize:
        xl, xr, match_view = match_result
    else:
        xl, xr = match_result
    print(
        f"PASS {img1_path.name} -> {img2_path.name}: "
        f"{len(kp1)} keypoints, {len(kp2)} keypoints, {len(xl)} matches"
    )
    odometry.form_epipolar_constraint(xr, xl)
    return match_view if visualize else None


def match_visualizations(pairs, max_display_matches=None):
    for img1_path, img2_path in pairs:
        yield run_pair(
            img1_path,
            img2_path,
            visualize=True,
            max_display_matches=max_display_matches,
        )

def main():
    args = parse_args()
    paths = args.images if args.images else discover_images(args.data_dir)
    pairs = image_pairs(paths)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    if args.visualize:
        show_image_stream(
            match_visualizations(pairs, args.max_display_matches),
            interval=args.interval,
        )
    else:
        for img1_path, img2_path in pairs:
            run_pair(img1_path, img2_path)

if __name__ == "__main__":
    main()
