#!/usr/bin/env python3
"""Load LeNet5 weights from a .wts file and export ONNX.

Expected .wts format:
- First line: number of parameter entries
- Each following line: <name> <count> <hex1> <hex2> ...
"""


import argparse
import struct
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def hex_to_float32(hex_str: str) -> float:
    """Convert one 32-bit hex token (e.g. 3f800000) into float32."""
    token = hex_str.lower().replace("0x", "")
    if len(token) != 8:
        raise ValueError(f"Invalid fp32 token length: {hex_str}")
    bits = int(token, 16)
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def load_wts(wts_path: Path) -> Dict[str, torch.Tensor]:
    """Parse .wts and return a dict of named float tensors (1D)."""
    lines = [line.strip() for line in wts_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty wts file: {wts_path}")

    expected_entries = int(lines[0])
    parsed: Dict[str, torch.Tensor] = {}

    for idx, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Malformed line {idx}: {line}")

        name = parts[0]
        count = int(parts[1])
        raw_vals = parts[2:]
        if len(raw_vals) != count:
            raise ValueError(
                f"Count mismatch at line {idx} for {name}: "
                f"declared={count}, actual={len(raw_vals)}"
            )

        values = [hex_to_float32(v) for v in raw_vals]
        parsed[name] = torch.tensor(values, dtype=torch.float32)

    if len(parsed) != expected_entries:
        raise ValueError(
            f"Entry count mismatch: header={expected_entries}, parsed={len(parsed)}"
        )

    return parsed


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


def assign_weights(model: LeNet5, weights: Dict[str, torch.Tensor]) -> None:
    """Map parsed 1D tensors into model parameter shapes and copy in-place."""
    state_dict = model.state_dict()

    missing = [name for name in state_dict.keys() if name not in weights]
    extra = [name for name in weights.keys() if name not in state_dict]
    if missing:
        raise KeyError(f"Missing weights in .wts: {missing}")
    if extra:
        raise KeyError(f"Unexpected weights in .wts: {extra}")

    remapped = {}
    for name, tensor in state_dict.items():
        flat = weights[name]
        numel = tensor.numel()
        if flat.numel() != numel:
            raise ValueError(
                f"Shape mismatch for {name}: expected {numel} values, got {flat.numel()}"
            )
        remapped[name] = flat.reshape(tensor.shape)

    model.load_state_dict(remapped, strict=True)


def validate_with_ones_input(model: LeNet5, batch: int) -> None:
    """Run one forward pass with all-ones input before ONNX export."""
    ones_input = torch.ones(batch, 1, 32, 32, dtype=torch.float32)
    with torch.no_grad():
        probs = model(ones_input)

    flat = probs.flatten()
    preview_count = min(5, flat.numel())
    preview_vals = [float(v) for v in flat[:preview_count]]

    print("Pre-export PyTorch validation:")
    print(f"  input shape:  {tuple(ones_input.shape)}")
    print(f"  output shape: {tuple(probs.shape)}")
    print(f"  output min/max: {float(probs.min()):.6f} / {float(probs.max()):.6f}")
    print(f"  output preview({preview_count}): {preview_vals}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LeNet5 ONNX from .wts")
    parser.add_argument(
        "--wts",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "lenet5.wts",
        help="Path to .wts file",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "lenet5.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--batch", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip pre-export PyTorch validation with all-ones input",
    )
    args = parser.parse_args()

    weights = load_wts(args.wts)

    model = LeNet5().eval()
    with torch.no_grad():
        assign_weights(model, weights)

    if not args.skip_validate:
        validate_with_ones_input(model, args.batch)

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(args.batch, 1, 32, 32, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        args.onnx.as_posix(),
        input_names=["input"],
        output_names=["probabilities"],
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Loaded weights from: {args.wts}")
    print(f"Exported ONNX to:   {args.onnx}")


if __name__ == "__main__":
    main()
