#!/usr/bin/env python3
"""Compare TensorRT `resnet_demo` output with PyTorch ResNet50 output."""

import argparse
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+")


def load_imagenet_classes(txt_path: Path) -> list[str]:
    classes = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(classes) < 1000:
        raise ValueError(f"Class file has too few entries: {len(classes)} < 1000 ({txt_path})")
    return classes


def preprocess_image(image_path: Path, width: int = 224, height: int = 224) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    f32 = rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    f32 = (f32 - mean) / std

    chw = np.transpose(f32, (2, 0, 1))
    return chw[np.newaxis, ...]


def run_trt_and_parse(binary_path: Path, image_path: Path) -> np.ndarray:
    result = subprocess.run(
        [str(binary_path), str(image_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    probs = []
    for line in result.stdout.splitlines():
        if line.startswith("Predicted class:"):
            break
        nums = [float(x) for x in FLOAT_RE.findall(line)]
        probs.extend(nums)

    if len(probs) < 1000:
        raise RuntimeError(
            "Failed to parse TensorRT probabilities from program output.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # 日志行可能包含时间戳/显存等数字，取末尾1000个更稳妥（概率通常在最后打印）。
    return np.array(probs[-1000:], dtype=np.float32)


def run_pytorch(input_nchw: np.ndarray) -> np.ndarray:
    if hasattr(torchvision.models, "ResNet50_Weights"):
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    else:
        model = torchvision.models.resnet50(pretrained=True)

    model = model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = torch.from_numpy(input_nchw).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    return probs.squeeze(0).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TensorRT and PyTorch ResNet50 outputs")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "build" / "src" / "resnet" / "resnet_demo",
        help="Path to resnet_demo binary",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "imagenet_classes.txt",
        help="Path to ImageNet class names txt",
    )
    args = parser.parse_args()

    classes = load_imagenet_classes(args.classes)
    input_nchw = preprocess_image(args.image)
    trt_probs = run_trt_and_parse(args.binary, args.image)
    torch_probs = run_pytorch(input_nchw)

    trt_top1 = int(np.argmax(trt_probs))
    torch_top1 = int(np.argmax(torch_probs))
    trt_name = classes[trt_top1] if 0 <= trt_top1 < len(classes) else "<out-of-range>"
    torch_name = classes[torch_top1] if 0 <= torch_top1 < len(classes) else "<out-of-range>"

    max_abs_diff = float(np.max(np.abs(trt_probs - torch_probs)))
    mean_abs_diff = float(np.mean(np.abs(trt_probs - torch_probs)))
    cosine = float(np.dot(trt_probs, torch_probs) / (np.linalg.norm(trt_probs) * np.linalg.norm(torch_probs) + 1e-12))

    print(f"TRT top1:   {trt_top1}")
    print(f"TRT class:  {trt_name}")
    print(f"Torch top1: {torch_top1}")
    print(f"Torch class:{torch_name}")
    print(f"Top1 same:  {trt_top1 == torch_top1}")
    print(f"Max abs diff:  {max_abs_diff:.6e}")
    print(f"Mean abs diff: {mean_abs_diff:.6e}")
    print(f"Cosine sim:    {cosine:.8f}")


if __name__ == "__main__":
    main()
