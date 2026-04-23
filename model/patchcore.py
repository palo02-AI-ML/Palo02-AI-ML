from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .backbone import FeatureExtractor
from .coreset import greedy_coreset_indices


class PatchCore:
    """PatchCore anomaly detection model.

    Paper: "Towards Total Recall in Industrial Anomaly Detection"
    (Roth et al., CVPR 2022).

    Pipeline:
        1. Extract multi-scale patch features from a frozen backbone.
        2. Apply neighborhood aggregation (avg-pool) on each feature map.
        3. Subsample with greedy k-center coreset to build a compact memory bank.
        4. At inference, score each patch by its distance to the nearest
           memory-bank entry; image score = max patch score.
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.patch_size: int = config["patchcore"]["patch_size"]
        self.coreset_ratio: float = config["patchcore"]["coreset_ratio"]
        self.max_candidates: int = config["patchcore"].get("max_candidates", 50_000)
        self.image_size: int = config["dataset"]["image_size"]
        self.memory_bank: Optional[torch.Tensor] = None

        self.extractor = FeatureExtractor(
            backbone_name=config["backbone"]["name"],
            layers=config["backbone"]["layers"],
        ).to(device)

    # ------------------------------------------------------------------
    # Internal feature extraction
    # ------------------------------------------------------------------

    def _extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return aggregated patch features of shape (B, C, H, W)."""
        feature_maps = self.extractor(images)
        layers = self.config["backbone"]["layers"]

        # Upsample all maps to the spatial size of the first (shallowest) layer
        target_size = feature_maps[layers[0]].shape[-2:]
        aligned = []
        for layer in layers:
            feat = feature_maps[layer]
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            aligned.append(feat)

        features = torch.cat(aligned, dim=1)  # (B, C_total, H, W)

        # Local neighborhood aggregation — captures spatial context
        features = F.avg_pool2d(
            features,
            kernel_size=self.patch_size,
            stride=1,
            padding=self.patch_size // 2,
        )
        return features  # (B, C_total, H, W)

    # ------------------------------------------------------------------
    # Training: build memory bank
    # ------------------------------------------------------------------

    def fit(self, dataloader) -> None:
        """Build the memory bank from normal training images."""
        all_patches = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                images = batch["image"].to(self.device)
                features = self._extract_patch_features(images)  # (B, C, H, W)
                B, C, H, W = features.shape
                patches = features.permute(0, 2, 3, 1).reshape(-1, C)
                all_patches.append(patches.cpu())

        all_patches = torch.cat(all_patches, dim=0)  # (N_total, C)
        print(
            f"Total patches: {all_patches.shape[0]:,} | "
            f"Dimensions: {all_patches.shape[1]}"
        )

        print("Applying greedy coreset subsampling…")
        indices = greedy_coreset_indices(
            all_patches.to(self.device),
            ratio=self.coreset_ratio,
            max_candidates=self.max_candidates,
        )
        self.memory_bank = all_patches[indices.cpu()].to(self.device)
        print(f"Memory bank size: {self.memory_bank.shape[0]:,} patches")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute anomaly scores and pixel-level anomaly maps.

        Args:
            images: (B, 3, H, W) normalized input images.

        Returns:
            image_scores: (B,) scalar anomaly score per image.
            anomaly_maps: (B, image_size, image_size) spatial anomaly heat map.
        """
        features = self._extract_patch_features(images)  # (B, C, H, W)
        B, C, H, W = features.shape

        patches = features.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)

        # Nearest-neighbor distance in the memory bank
        dists = torch.cdist(patches.float(), self.memory_bank.float())  # (B*H*W, M)
        nn_dists, _ = dists.min(dim=1)  # (B*H*W,)

        patch_scores = nn_dists.reshape(B, H, W)

        anomaly_maps = F.interpolate(
            patch_scores.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, image_size, image_size)

        image_scores = anomaly_maps.flatten(1).max(dim=1).values  # (B,)
        return image_scores, anomaly_maps

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"memory_bank": self.memory_bank, "config": self.config}, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> "PatchCore":
        data = torch.load(path, map_location=device)
        model = cls(data["config"], device)
        model.memory_bank = data["memory_bank"]
        return model
