from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MVTecDataset(Dataset):
    """MVTec Anomaly Detection dataset loader.

    Directory structure expected:
        <root>/<category>/train/good/*.png
        <root>/<category>/test/<defect_type>/*.png
        <root>/<category>/ground_truth/<defect_type>/*_mask.png
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        image_size: int = 224,
    ):
        self.root = Path(root) / category
        self.split = split
        self.image_size = image_size

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        split_dir = self.root / self.split

        if self.split == "train":
            for path in sorted((split_dir / "good").glob("*.png")):
                samples.append({"image": path, "mask": None, "label": 0, "defect": "good"})
        else:
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect = defect_dir.name
                label = 0 if defect == "good" else 1
                mask_dir = self.root / "ground_truth" / defect if defect != "good" else None

                for path in sorted(defect_dir.glob("*.png")):
                    mask_path: Optional[Path] = None
                    if mask_dir is not None:
                        candidate = mask_dir / (path.stem + "_mask.png")
                        mask_path = candidate if candidate.exists() else None

                    samples.append({
                        "image": path,
                        "mask": mask_path,
                        "label": label,
                        "defect": defect,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = Image.open(sample["image"]).convert("RGB")
        image_tensor = self.image_transform(image)

        if sample["mask"] is not None:
            mask = Image.open(sample["mask"]).convert("L")
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0).float()
        else:
            mask_tensor = torch.zeros(1, self.image_size, self.image_size)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": sample["label"],
            "defect": sample["defect"],
            "path": str(sample["image"]),
        }
