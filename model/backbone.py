from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models

from .gabor_backbone import GaborFeatureExtractor


class FeatureExtractor(nn.Module):
    """Feature extractor with two backends:
      - 'gabor': Gabor filter bank (no download needed, great for textures)
      - any torchvision model name: pretrained CNN backbone
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
    ):
        super().__init__()
        self.target_layers = layers or ["layer2", "layer3"]
        self._outputs: Dict[str, torch.Tensor] = {}
        self._use_gabor = (backbone_name == "gabor")

        if self._use_gabor:
            self._gabor = GaborFeatureExtractor(orientations=8)
        else:
            backbone = self._load_pretrained(backbone_name)
            for name, module in backbone.named_children():
                if name in self.target_layers:
                    module.register_forward_hook(self._make_hook(name))
            self.backbone = backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    @staticmethod
    def _load_pretrained(name: str) -> nn.Module:
        weights_map = {
            "wide_resnet50_2": "Wide_ResNet50_2_Weights",
            "resnet50": "ResNet50_Weights",
            "resnet18": "ResNet18_Weights",
        }
        try:
            weights_cls = getattr(models, weights_map[name])
            return getattr(models, name)(weights=weights_cls.IMAGENET1K_V1)
        except (AttributeError, KeyError):
            return getattr(models, name)(pretrained=True)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._outputs[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._use_gabor:
            return self._gabor(x)
        self._outputs = {}
        self.backbone(x)
        return {name: self._outputs[name] for name in self.target_layers}
