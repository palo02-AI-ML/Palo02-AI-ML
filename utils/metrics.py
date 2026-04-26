from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalized (C, H, W) tensor to a uint8 HWC array."""
    img = tensor.cpu() * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def compute_image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def compute_pixel_auroc(masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    return float(roc_auc_score(masks.flatten().astype(int), anomaly_maps.flatten()))


def save_anomaly_maps(
    images: List[torch.Tensor],
    masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    labels: List[int],
    defects: List[str],
    output_dir: str,
    max_samples: int = 16,
) -> None:
    """Save side-by-side visualizations: image | GT mask | anomaly map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(images), max_samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None]

    col_titles = ["Image", "Ground Truth", "Anomaly Map"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        img_np = denormalize(images[i])
        mask_np = masks[i]
        amap = anomaly_maps[i]

        vmin, vmax = amap.min(), amap.max()
        amap_norm = (amap - vmin) / (vmax - vmin + 1e-8)

        label_str = f"{'ANOMALY' if labels[i] else 'NORMAL'} ({defects[i]})"

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(label_str, fontsize=8)

        axes[i, 1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)

        im = axes[i, 2].imshow(img_np)
        axes[i, 2].imshow(amap_norm, cmap="jet", alpha=0.5, vmin=0, vmax=1)

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    out_path = output_dir / "anomaly_maps.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization → {out_path}")


def plot_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve → {output_path}")
