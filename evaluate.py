"""Evaluate PatchCore: compute AUROC metrics and save visualizations."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.mvtec import MVTecDataset
from model.patchcore import PatchCore
from utils.metrics import (
    compute_image_auroc,
    compute_pixel_auroc,
    plot_roc_curve,
    save_anomaly_maps,
)


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"].get("device", "auto"))
    print(f"Device: {device}")

    memory_bank_path = config["output"]["memory_bank_path"]
    model = PatchCore.load(memory_bank_path, device)
    print(f"Memory bank loaded: {model.memory_bank.shape[0]:,} patches")

    dataset = MVTecDataset(
        root=config["dataset"]["data_path"],
        category=config["dataset"]["category"],
        split="test",
        image_size=config["dataset"]["image_size"],
    )
    print(f"Test images: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=8, num_workers=2, shuffle=False)

    all_scores, all_labels = [], []
    all_maps, all_masks = [], []
    sample_images, sample_defects = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)

        scores, maps = model.predict(images)

        all_scores.extend(scores.cpu().numpy())
        all_labels.extend(batch["label"].numpy())
        all_maps.extend(maps.cpu().numpy())
        all_masks.extend(batch["mask"].squeeze(1).numpy())
        sample_images.extend(list(batch["image"]))
        sample_defects.extend(batch["defect"])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_maps = np.array(all_maps)
    all_masks = np.array(all_masks)

    results_dir = Path(config["output"]["results_path"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Image-level AUROC
    image_auroc = compute_image_auroc(all_labels, all_scores)
    print(f"\n{'='*40}")
    print(f"Image AUROC : {image_auroc:.4f}")

    plot_roc_curve(
        all_labels,
        all_scores,
        title=f"Image AUROC — {config['dataset']['category']}",
        output_path=str(results_dir / "roc_image.png"),
    )

    # Pixel-level AUROC (only meaningful when anomalous masks exist)
    has_defect = all_labels == 1
    if has_defect.any() and all_masks[has_defect].sum() > 0:
        pixel_auroc = compute_pixel_auroc(all_masks, all_maps)
        print(f"Pixel AUROC : {pixel_auroc:.4f}")

        plot_roc_curve(
            all_masks.flatten().astype(int),
            all_maps.flatten(),
            title=f"Pixel AUROC — {config['dataset']['category']}",
            output_path=str(results_dir / "roc_pixel.png"),
        )
    else:
        print("Pixel AUROC : N/A (no mask annotations found)")

    print(f"{'='*40}\n")

    # Pick samples to visualize: anomalous first, then normal
    anomaly_idx = np.where(all_labels == 1)[0].tolist()
    normal_idx = np.where(all_labels == 0)[0].tolist()
    viz_idx = (anomaly_idx + normal_idx)[:16]

    save_anomaly_maps(
        images=[sample_images[i] for i in viz_idx],
        masks=[all_masks[i] for i in viz_idx],
        anomaly_maps=[all_maps[i] for i in viz_idx],
        labels=[int(all_labels[i]) for i in viz_idx],
        defects=[sample_defects[i] for i in viz_idx],
        output_dir=str(results_dir),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PatchCore")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
