"""Build the PatchCore memory bank from normal training images."""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.mvtec import MVTecDataset
from model.patchcore import PatchCore


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
    print(f"Category: {config['dataset']['category']}")

    dataset = MVTecDataset(
        root=config["dataset"]["data_path"],
        category=config["dataset"]["category"],
        split="train",
        image_size=config["dataset"]["image_size"],
    )
    print(f"Training images: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model = PatchCore(config, device)
    model.fit(dataloader)

    out_path = Path(config["output"]["memory_bank_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PatchCore memory bank")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
