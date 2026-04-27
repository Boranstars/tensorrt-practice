#!/usr/bin/env python3
"""Export pretrained GoogLeNet to ONNX.

Default output: src/googlenet/models/googlenet.onnx
"""

import argparse
from pathlib import Path

import torch
import torchvision


def build_model() -> torch.nn.Module:
    """Create a pretrained GoogLeNet model in eval mode."""
    # torchvision>=0.13 prefers weights over pretrained.
    if hasattr(torchvision.models, "GoogLeNet_Weights"):
        model = torchvision.models.googlenet(
            weights=torchvision.models.GoogLeNet_Weights.DEFAULT,
        )
    else:
        model = torchvision.models.googlenet(pretrained=True)

    # Some torchvision versions require aux_logits=True when loading pretrained
    # weights. Turn off aux branches afterward for cleaner inference/export setup.
    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None

    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pretrained GoogLeNet to ONNX")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "googlenet.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--batch", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force export on CPU even when CUDA is available",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Export with dynamic batch dimension",
    )
    args = parser.parse_args()

    cuda_count = torch.cuda.device_count()
    print(f"cuda device count: {cuda_count}")

    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"using device: {device}")

    net = build_model().to(device)
    print(net)

    dummy_input = torch.ones(args.batch, 3, 224, 224, device=device)
    with torch.no_grad():
        out = net(dummy_input)
    print(f"googlenet out: {tuple(out.shape)}")

    args.onnx.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        net,
        dummy_input,
        args.onnx.as_posix(),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"Exported ONNX to: {args.onnx}")


if __name__ == "__main__":
    main()
