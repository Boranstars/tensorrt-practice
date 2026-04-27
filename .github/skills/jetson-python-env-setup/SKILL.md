---
name: jetson-python-env-setup
description: '在 Jetson 上用 uv 构建可复现 Python 环境，并按 NVIDIA 官方文档与设备信息安装匹配的 PyTorch/torchvision。适用于 JetPack/L4T 版本不确定、索引选择、安装失败排查、CUDA 可用性验证。'
argument-hint: 'Jetson Python 环境目标（例如：JP6.1 安装并验证 torch+torchvision）'
---

# Jetson Python Environment Setup

## 适用场景

- 需要在 Jetson 上搭建可复现的 Python 环境。
- 需要根据 JetPack/L4T 与官方可用索引安装正确的 PyTorch。
- 安装后需要验证 CUDA 是否生效。

## 产出目标

- 创建并同步 `.venv` 环境（uv 管理）。
- 完成与设备匹配的 torch 安装（必要时处理 wheel 文件名校验问题）。
- 运行验证脚本并给出是否通过的结论。

## 参考资源

- 设备与索引探测脚本: [jetson_pytorch_probe.sh](./scripts/jetson_pytorch_probe.sh)
- 安装验证脚本: [verify_torch_cuda.py](./scripts/verify_torch_cuda.py)
- NVIDIA 安装指南: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- NVIDIA 兼容矩阵: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html

## 标准流程

1. 前置检查

```bash
bash .github/skills/jetson-python-env-setup/scripts/jetson_pytorch_probe.sh
```

检查点：

- 架构应为 `aarch64`。
- Python 与 `.venv` 版本符合项目要求（通常 3.10）。
- 确认 `nvidia-l4t-core` 版本。
- 在 NVIDIA `jp/vXX/pytorch/` 中选择可访问（HTTP 200）且包含 `cp310` wheel 的最新目录。

2. 初始化环境

```bash
uv venv --python 3.10
uv sync --extra export --extra jetson
```

3. 安装 torch（按设备匹配的 NVIDIA 官方 wheel）

```bash
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python .venv/bin/python "<NVIDIA_TORCH_WHEEL_URL>"
```

说明：

- 某些 NVIDIA wheel 可能触发 uv 的文件名版本严格校验；加 `UV_SKIP_WHEEL_FILENAME_CHECK=1` 可绕过该问题。
- 推荐每次先 `uv sync`，再安装 torch/torchvision，避免 sync 移除手工安装包。

4. 安装 torchvision
torchvision 在jetson上可能没有官方 wheel，建议手动编译或者非必须则不安装。如果需要安装，确保版本与 torch 兼容。
推荐优先使用源码安装到当前 `.venv`，并禁止依赖重解：

先准备编译依赖（仅需执行一次）：

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake pkg-config \
  libjpeg-dev libpng-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev
```

再克隆并切换到匹配版本（以 torch 2.5 对应 0.20.x 为例）：

```bash

git clone --branch v0.20.0 --depth 1 https://github.com/pytorch/vision.git /tmp/vision-src
```

最后在当前虚拟环境中编译安装：

```bash
cd /tmp/vision-src
MAX_JOBS=4 /path/to/repo/.venv/bin/python -m pip install -v --no-deps --no-build-isolation .
```

推荐原因：

- Jetson 上经常缺少可直接匹配的 torchvision 官方 wheel。
- 直接 `uv pip install torchvision` 往往会重解依赖并替换已安装的 Jetson torch（破坏已验证环境）。
- `--no-deps --no-build-isolation` 可避免构建阶段重新安装/替换 torch。

选择规则：

- torchvision 源码分支/标签应与当前 torch 主版本兼容（例如 torch 2.5 对应 torchvision 0.20.x）。
- 优先使用本机已验证可编译通过的源码版本。

5. 验证安装

```bash
.venv/bin/python .github/skills/jetson-python-env-setup/scripts/verify_torch_cuda.py
```

通过标准：

- `cuda_available` 为 `True`
- `cuda_version` 非空
- `device_count >= 1`
- （可选）`torchvision` 非 `not-installed`（如果你的任务依赖 torchvision）

## 决策分支

- 若 `jp/v62` 等目标目录返回 404：
  使用可访问且有对应 wheel 的最近目录（例如 `jp/v61`），并在记录中注明原因。

- 若出现 NumPy ABI 警告（NumPy 2.x 与 Jetson torch 轮子不兼容）：
  在项目依赖中固定 `numpy<2`，再执行 `uv sync`。

- 若 `uv sync` 后 torch 被移除：
  按顺序重做：`uv sync` -> 安装 torch  -> 验证。

- 若安装 torchvision 时“卡在 preparing packages”：
  通常是依赖重解、下载或 build isolation 导致。改用源码安装命令并加 `--no-deps --no-build-isolation`。

- 若源码编译慢：
  Jetson 编译 C++/CUDA 扩展耗时较长属正常，建议设置 `MAX_JOBS` 控制并行度（如 2-4）。

- 若 `uv run` 导致环境漂移：
  排查期优先使用 `.venv/bin/python` 直接执行验证脚本。

## 完成检查清单

- 已输出探测结果（设备、L4T、可用索引、wheel 线索）。
- 已完成 torch 安装，且记录来源 URL。
- 已完成验证脚本执行，并给出 CUDA 验证结果。
- 已说明复现命令顺序，避免环境漂移。
