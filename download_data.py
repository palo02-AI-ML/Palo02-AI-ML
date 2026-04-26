"""Download and extract the MVTec AD carpet category.

Usage:
    python download_data.py [--category carpet] [--dest ./data/mvtec]

The MVTec AD dataset requires accepting the research license at:
    https://www.mvtec.com/company/research/datasets/mvtec-ad

This script downloads the full archive (~4.9 GB) and extracts
only the requested category.
"""

import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

DATASET_URL = (
    "https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/"
    "company/research/datasets/mvtec_anomaly_detection.tar.xz"
)


def download(url: str, dest: Path) -> Path:
    archive = dest / "mvtec_anomaly_detection.tar.xz"
    if archive.exists():
        print(f"Archive already present: {archive}")
        return archive

    print(f"Downloading {url}")
    print("(~4.9 GB — this may take a while)\n")

    try:
        subprocess.run(
            ["wget", "-c", "-O", str(archive), url],
            check=True,
        )
    except FileNotFoundError:
        # wget not available — try curl
        subprocess.run(
            ["curl", "-L", "-C", "-", "-o", str(archive), url],
            check=True,
        )
    return archive


def extract_category(archive: Path, category: str, dest: Path) -> None:
    out_dir = dest / category
    if out_dir.exists():
        print(f"Category already extracted: {out_dir}")
        return

    print(f"Extracting '{category}' from archive…")
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:xz") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith(category + "/")]
        if not members:
            print(f"ERROR: category '{category}' not found in archive.", file=sys.stderr)
            sys.exit(1)
        tf.extractall(path=dest, members=members)

    print(f"Extracted → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument("--category", default="carpet", help="Category to extract")
    parser.add_argument("--dest", default="./data/mvtec", help="Output directory")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    archive = download(DATASET_URL, dest)
    extract_category(archive, args.category, dest)

    print("\nDone. Update config.yaml if you chose a different category.")


if __name__ == "__main__":
    main()
