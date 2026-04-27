#!/usr/bin/env python3
import importlib.util


def _safe_pkg_version(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not-installed"
    mod = __import__(name)
    return getattr(mod, "__version__", "unknown")


def main() -> None:
    import torch

    print("torch", torch.__version__)
    print("torchvision", _safe_pkg_version("torchvision"))
    print("cuda_available", torch.cuda.is_available())
    print("cuda_version", torch.version.cuda)
    print("cudnn_version", torch.backends.cudnn.version())
    print("device_count", torch.cuda.device_count())


if __name__ == "__main__":
    main()
