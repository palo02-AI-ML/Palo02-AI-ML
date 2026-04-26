"""Gradio demo for PatchCore anomaly detection.

Usage:
    python app.py                         # loads config.yaml and outputs/memory_bank.pt
    python app.py --config my_config.yaml
"""

import argparse
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

from model.patchcore import PatchCore

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(config_path: str) -> tuple[PatchCore, dict, torch.device]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    model = PatchCore.load(config["output"]["memory_bank_path"], device)
    print(f"Model loaded | Memory bank: {model.memory_bank.shape[0]:,} patches | Device: {device}")
    return model, config, device


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    return IMAGE_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)


def build_overlay(
    original: Image.Image,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
) -> Image.Image:
    """Blend the jet heatmap over the original image."""
    h, w = original.size[1], original.size[0]
    norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heatmap = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap).resize((w, h), Image.BILINEAR)
    return Image.blend(original.convert("RGB"), heatmap_pil, alpha=alpha)


def build_heatmap_only(anomaly_map: np.ndarray, size: tuple) -> Image.Image:
    norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    heatmap = (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(heatmap).resize(size, Image.BILINEAR)


def score_to_label(score: float, threshold: float) -> str:
    if score >= threshold:
        return f"🔴 ANOMALÍA  —  score: {score:.4f}"
    return f"🟢 NORMAL  —  score: {score:.4f}"


# ---------------------------------------------------------------------------
# Calibration: compute a threshold from the memory bank
# We use the 95th percentile of training-set self-distances as a proxy.
# ---------------------------------------------------------------------------

def estimate_threshold(model: PatchCore) -> float:
    bank = model.memory_bank.float()
    # Sample 2000 patches to keep it fast
    idx = torch.randperm(len(bank))[:2000]
    sample = bank[idx]
    dists = torch.cdist(sample, bank).min(dim=1).values
    return float(dists.quantile(0.95).item())


# ---------------------------------------------------------------------------
# Main inference function (called by Gradio)
# ---------------------------------------------------------------------------

def run_inference(pil_image, alpha_slider):
    if pil_image is None:
        return None, None, "Sin imagen"

    tensor = preprocess(pil_image).to(DEVICE)
    score_tensor, map_tensor = MODEL.predict(tensor)

    score = float(score_tensor[0].item())
    amap = map_tensor[0].cpu().numpy()

    orig_size = (pil_image.size[0], pil_image.size[1])

    overlay = build_overlay(pil_image, amap, alpha=alpha_slider)
    heatmap = build_heatmap_only(amap, orig_size)
    label = score_to_label(score, THRESHOLD)

    return overlay, heatmap, label


# ---------------------------------------------------------------------------
# Sample images helper
# ---------------------------------------------------------------------------

def get_sample_paths(config: dict) -> dict[str, list[str]]:
    data_root = Path(config["dataset"]["data_path"]) / config["dataset"]["category"]
    samples = {}
    for split_dir in ["test/good", "test/cut", "test/color"]:
        key = split_dir.split("/")[-1]
        d = data_root / split_dir
        if d.exists():
            imgs = sorted(d.glob("*.png"))[:4]
            samples[key] = [str(p) for p in imgs]
    return samples


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui(sample_paths: dict) -> gr.Blocks:
    with gr.Blocks(title="PatchCore - Deteccion de Anomalias") as demo:

        gr.Markdown(
            """
            # 🔍 PatchCore — Detección de Anomalías en Fibra de Carbono
            Sube una imagen o elige un ejemplo. El modelo detecta defectos sin haber visto
            ninguno durante el entrenamiento.
            """
        )

        with gr.Row():
            # --- Inputs ---
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Imagen de entrada",
                    height=300,
                )
                alpha_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Intensidad del mapa de calor",
                )
                run_btn = gr.Button("▶  Analizar", variant="primary")

                gr.Markdown("### Ejemplos")
                with gr.Tabs():
                    for label, paths in sample_paths.items():
                        with gr.Tab(label):
                            gr.Examples(
                                examples=paths,
                                inputs=image_input,
                                label="",
                            )

            # --- Outputs ---
            with gr.Column(scale=1):
                result_label = gr.Textbox(
                    label="Resultado",
                    interactive=False,
                    lines=1,
                )
                overlay_output = gr.Image(
                    label="Imagen + mapa de calor",
                    height=300,
                )
                heatmap_output = gr.Image(
                    label="Mapa de anomalía (jet)",
                    height=300,
                )

        run_btn.click(
            fn=run_inference,
            inputs=[image_input, alpha_slider],
            outputs=[overlay_output, heatmap_output, result_label],
        )
        image_input.change(
            fn=run_inference,
            inputs=[image_input, alpha_slider],
            outputs=[overlay_output, heatmap_output, result_label],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio link")
    args = parser.parse_args()

    MODEL, CONFIG, DEVICE = load_model(args.config)
    THRESHOLD = estimate_threshold(MODEL)
    print(f"Anomaly threshold (p95): {THRESHOLD:.4f}")

    samples = get_sample_paths(CONFIG)
    demo = build_ui(samples)
    demo.launch(share=args.share, theme=gr.themes.Soft())
