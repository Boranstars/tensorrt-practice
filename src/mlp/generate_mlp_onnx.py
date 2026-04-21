#!/usr/bin/env python3
"""Generate a tiny MLP ONNX model with fixed parameters.

Given parameters:
2
linear.weight 1 3fff7e32
linear.bias 1 3c138a5a
"""



import struct
from pathlib import Path

import torch
import torch.nn as nn


def hex_to_float32(hex_str: str) -> float:
    """Convert a hex bit-pattern (e.g. 3f800000) to float32."""
    hex_str = hex_str.lower().replace("0x", "")
    if len(hex_str) != 8:
        raise ValueError(f"Expected 8 hex chars for float32, got: {hex_str}")
    as_int = int(hex_str, 16)
    return struct.unpack("!f", struct.pack("!I", as_int))[0]


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> None:
    # Parameters copied from user input.
    weight_hex = "3fff7e32"
    bias_hex = "3c138a5a"

    weight = hex_to_float32(weight_hex)
    bias = hex_to_float32(bias_hex)

    model = TinyMLP().eval()

    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[weight]], dtype=torch.float32))
        model.linear.bias.copy_(torch.tensor([bias], dtype=torch.float32))

    module_dir = Path(__file__).resolve().parent
    models_dir = module_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = models_dir / "mlp.onnx"
    dummy_input = torch.randn(1, 1, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        
    )

    print(f"Exported ONNX: {onnx_path}")
    print(f"linear.weight = {weight}")
    print(f"linear.bias   = {bias}")


if __name__ == "__main__":
    main()
