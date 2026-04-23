"""Generate a synthetic carbon-fiber-like dataset in MVTec AD format.

Creates a woven texture (normal) and injects artificial defects (anomalies)
so the full PatchCore pipeline can be exercised without downloading MVTec.

Output structure mirrors MVTec AD:
  data/mvtec/carpet/
    train/good/            ← normal images for memory bank
    test/good/             ← normal test images
    test/cut/              ← simulated cut defects
    test/color/            ← simulated discolouration defects
    ground_truth/cut/      ← binary masks
    ground_truth/color/
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Texture generation
# ---------------------------------------------------------------------------

def make_fiber_texture(size: int = 256, noise_std: float = 12.0) -> np.ndarray:
    """Create a woven carbon-fiber-like RGB texture."""
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    # Two diagonal weave directions
    freq = 2 * np.pi / 16
    wave1 = np.sin(freq * (xx + yy))
    wave2 = np.sin(freq * (xx - yy))

    # Combine and scale to [0, 1]
    pattern = 0.5 * (wave1 + wave2)
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())

    # Dark base colour (carbon fiber is almost black)
    base = (pattern * 60 + 15).astype(np.float32)      # ~15-75 dark gray

    # Slight blue-grey tint
    r = (base * 0.90 + RNG.normal(0, noise_std, (size, size))).clip(0, 255)
    g = (base * 0.92 + RNG.normal(0, noise_std, (size, size))).clip(0, 255)
    b = (base * 1.00 + RNG.normal(0, noise_std, (size, size))).clip(0, 255)

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Defect injection
# ---------------------------------------------------------------------------

def inject_cut(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add a bright diagonal scratch to simulate a cut fiber."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x0 = RNG.integers(20, w - 60)
    y0 = RNG.integers(20, h - 60)
    length = RNG.integers(40, 90)
    thickness = RNG.integers(3, 7)

    aug = img.copy().astype(np.float32)
    for i in range(length):
        px, py = x0 + i, y0 + i
        for t in range(-thickness, thickness + 1):
            cx, cy = px + t, py
            if 0 <= cx < w and 0 <= cy < h:
                aug[cy, cx] = [220, 210, 200]
                mask[cy, cx] = 255

    return aug.clip(0, 255).astype(np.uint8), mask


def inject_color(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add a resin-rich bright ellipse to simulate a colour defect."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx = RNG.integers(40, w - 40)
    cy = RNG.integers(40, h - 40)
    rx = RNG.integers(15, 35)
    ry = RNG.integers(10, 25)

    aug = img.copy().astype(np.float32)
    yy, xx = np.ogrid[:h, :w]
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1

    # Yellowish resin spot
    aug[ellipse, 0] = (aug[ellipse, 0] * 0.4 + 160).clip(0, 255)
    aug[ellipse, 1] = (aug[ellipse, 1] * 0.4 + 140).clip(0, 255)
    aug[ellipse, 2] = (aug[ellipse, 2] * 0.3 + 60).clip(0, 255)
    mask[ellipse] = 255

    return aug.clip(0, 255).astype(np.uint8), mask


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(root: Path, category: str, n_train: int, n_test_good: int, n_defects: int) -> None:
    cat = root / category
    dirs = [
        cat / "train" / "good",
        cat / "test" / "good",
        cat / "test" / "cut",
        cat / "test" / "color",
        cat / "ground_truth" / "cut",
        cat / "ground_truth" / "color",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    size = 224

    print(f"Generating {n_train} training images…")
    for i in range(n_train):
        img = make_fiber_texture(size)
        Image.fromarray(img).save(cat / "train" / "good" / f"{i:03d}.png")

    print(f"Generating {n_test_good} normal test images…")
    for i in range(n_test_good):
        img = make_fiber_texture(size)
        Image.fromarray(img).save(cat / "test" / "good" / f"{i:03d}.png")

    print(f"Generating {n_defects} cut defects…")
    for i in range(n_defects):
        img = make_fiber_texture(size)
        aug, mask = inject_cut(img)
        Image.fromarray(aug).save(cat / "test" / "cut" / f"{i:03d}.png")
        Image.fromarray(mask).save(cat / "ground_truth" / "cut" / f"{i:03d}_mask.png")

    print(f"Generating {n_defects} colour defects…")
    for i in range(n_defects):
        img = make_fiber_texture(size)
        aug, mask = inject_color(img)
        Image.fromarray(aug).save(cat / "test" / "color" / f"{i:03d}.png")
        Image.fromarray(mask).save(cat / "ground_truth" / "color" / f"{i:03d}_mask.png")

    total_test = n_test_good + 2 * n_defects
    print(f"\nDataset ready at {cat}")
    print(f"  Train : {n_train} images")
    print(f"  Test  : {total_test} images ({n_test_good} normal, {2*n_defects} anomalous)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic carbon-fiber dataset")
    parser.add_argument("--dest", default="./data/mvtec")
    parser.add_argument("--category", default="carpet")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test-good", type=int, default=40)
    parser.add_argument("--n-defects", type=int, default=30)
    args = parser.parse_args()

    build_dataset(
        root=Path(args.dest),
        category=args.category,
        n_train=args.n_train,
        n_test_good=args.n_test_good,
        n_defects=args.n_defects,
    )


if __name__ == "__main__":
    main()
