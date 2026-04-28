#!/usr/bin/env python3
"""Export pretrained ResNet50 to ONNX.

Default output: src/resnet/models/resnet.onnx
"""

from pathlib import Path

import torch
import torchvision


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    onnx_path = module_dir / "models" / "resnet.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(torchvision.models, "ResNet50_Weights"):
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT,
        )
    else:
        model = torchvision.models.resnet50(pretrained=True)

    model = model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_input = torch.ones(1, 3, 224, 224, device=device)

    with torch.no_grad():
        output = model(dummy_input)
    print(f"device: {device}")
    print(f"resnet50 out shape: {tuple(output.shape)}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
    )

    print(f"Exported ONNX to: {onnx_path}")


if __name__ == "__main__":
    main()
