# Python 环境配置（Jetson + uv）

本文档目标：让本仓库 Python 环境可复现，并在 Jetson 上正确安装 CUDA 版 PyTorch。

## 1. 原则

- 使用 uv 管理虚拟环境与依赖。
- 本仓库不作为 Python 包发布，仅做环境管理。
- Jetson 上的 torch/torchvision 必须按设备与 JetPack 匹配安装，不能盲目从 PyPI 安装通用 CPU 轮子。

## 2. 基础环境安装

在 Jetson 上先安装系统依赖（来自 NVIDIA 官方文档）：

```bash
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev
```

可选：若官方文档对应版本要求 cusparselt（24.06+ 常见），先按官方说明安装。

## 3. 创建项目虚拟环境

在仓库根目录执行：

```bash
uv venv --python 3.10
uv sync --extra export --extra jetson
```

说明：

- `uv sync` 会按 pyproject.toml 安装项目基础依赖。
- 本项目已限制 `numpy<2`，避免与部分 Jetson torch 轮子 ABI 不兼容。

## 4. 如何从官网选择正确的 PyTorch 安装地址

官方依据：

- 安装指南：
  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- 兼容矩阵（PyTorch 版本与 JetPack 对应关系）：
  https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html

官方 URL 模板：

```text
https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION/pytorch/$PYT_WHEEL
```

注意：

- 先根据设备 JetPack/L4T 与兼容矩阵确定目标版本。
- 再以“目录可访问且存在对应 cp310 wheel”为最终可用依据。
- 示例实测：本设备 `nvidia-l4t-core=36.4.3`，`jp/v62` 返回 404，`jp/v61` 可访问并有 cp310 wheel。

## 5. 一键提取设备信息与可用 index

运行脚本：

```bash
bash scripts/jetson_pytorch_probe.sh
```

脚本会输出：

- 架构、Python、uv 版本
- nvidia-l4t-core / nvidia-jetpack
- 官方 PyTorch 目录是否可达（HTTP 状态）
- 可用目录中的 cp310 wheel 文件名样例

## 6. 安装 Jetson 对应 torch

示例（本机实测可用）：

```bash
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python .venv/bin/python \
  "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
```

为什么加 `UV_SKIP_WHEEL_FILENAME_CHECK=1`：

- 某些 NVIDIA wheel 文件名版本与内部元数据字符串有细微差异，uv 会严格校验并报错。
- 该变量用于跳过文件名校验，属于已知兼容处理。

## 7. 安装 torchvision

原则：torchvision 也要与 torch 和 JetPack 匹配。

推荐顺序：

1. 先安装 torch（如上）。
2. 再安装匹配的 torchvision（Jetson 上优先源码安装，避免替换 torch）。
3. 安装后执行第 8 节验证。



更推荐的源码安装命令（已验证可用）：

先安装源码编译依赖（仅需执行一次）：

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential cmake pkg-config \
  libjpeg-dev libpng-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev
```

克隆 torchvision 源码并切换到匹配版本（示例为 torch 2.5 对应的 0.20.x）：

```bash
git clone --branch v0.20.0 --depth 1 https://github.com/pytorch/vision.git /tmp/vision-src
```

在当前 `.venv` 中编译安装：

```bash
cd /tmp/vision-src
MAX_JOBS=4 /home/jetson/Programs/tensorrt/tensorrt-practice/.venv/bin/python -m pip install -v --no-deps --no-build-isolation .
```

为什么推荐源码命令：

- 在当前 Jetson 场景下，直接 `uv pip install torchvision` 很可能触发依赖重解并替换已安装的 Jetson torch。
- `--no-deps --no-build-isolation` 可避免构建阶段重新解析依赖与构建隔离环境，减少环境漂移。

常见坑点：

- 坑 1：安装阶段长时间显示 “preparing packages”。
  常见原因是依赖重解、网络下载或 build isolation；优先改用上述源码命令。

- 坑 2：执行 `uv sync` 后手工安装的 torch/torchvision 被移除。
  解决方式：固定顺序为 `uv sync` -> 安装 torch -> 安装 torchvision -> 验证。

- 坑 3：构建非常慢看起来像卡住。
  Jetson 编译扩展耗时本就较长，可通过 `MAX_JOBS=2` 或 `MAX_JOBS=4` 控制并行度。

- 坑 5：源码版本与 torch 主版本不匹配导致导入或运行时报错。
  解决方式：按兼容关系选择 torchvision 版本（例如 torch 2.5 对应 0.20.x）。

- 坑 4：排查时使用 `uv run` 导致隐式同步。
  排查阶段优先用 `.venv/bin/python` 直接运行脚本。

## 8. 验证安装

运行验证脚本：

```bash
.venv/bin/python scripts/verify_torch_cuda.py
```

也可使用：

```bash
uv run python scripts/verify_torch_cuda.py
```

但在排查环境问题时，优先使用 `.venv/bin/python`，避免 `uv run` 在锁文件与环境不一致时触发隐式同步。

脚本输出以下关键字段：

- torch 版本
- torchvision 版本
- cuda_available
- cuda_version
- cudnn_version
- device_count

最小通过标准：

- `cuda_available` 为 `True`
- `cuda_version` 非空
- `device_count >= 1`

## 9. 可复现执行顺序

建议每次都按以下顺序：

1. `uv sync --extra export --extra jetson`
2. `UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install --python .venv/bin/python <torch_whl_url>`
3. 安装匹配的 torchvision
4. `uv run python scripts/verify_torch_cuda.py`

注意：

- `uv sync` 可能移除手工安装的未声明包。
- 因此 torch/torchvision 建议放在 `uv sync` 之后安装。
