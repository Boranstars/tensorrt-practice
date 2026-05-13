# tensorrt-practice 代理指南

## 目标
本工作区用于学习 TensorRT 推理，包含多个子模块（mlp、lenet、googlenet、resnet、yolov5 等），以及面向特定 API 的实验。

代理在执行任务时，应优先保证 CMake 组织可维护、可扩展，便于持续新增学习模块。

## 当前项目事实
- 构建系统：CMake + Ninja（build 目录中已存在 Ninja 构建文件）。
- 根构建文件：CMakeLists.txt。
- 目标环境：Linux，常见为 Jetson aarch64。
- 活跃 C++ 学习模块（均位于 src/ 下）：mlp、lenet、googlenet、resnet、yolov5。
  - 每个模块有独立的 CMakeLists.txt，通过 src/CMakeLists.txt 的 add_subdirectory 注册。
- Python 工具链：uv 管理虚拟环境，pyproject.toml 声明依赖，详见 docs/python-environment.md。
  - 项目不作为 Python 包发布。
- 文档：docs/ 目录。
- 辅助脚本：scripts/ 。

## 构建命令
除非用户另有说明，默认使用：

- 配置：cmake -S . -B build -G Ninja
- 构建：cmake --build build

如果做了结构调整后出现 CMake 缓存不一致问题，清理 build 并重新配置。


## CMake 组织规则
1. 顶层 CMakeLists.txt 只保留共享项目配置与共享依赖
2. 每个学习主题使用独立目录，并在该目录内维护自己的 CMakeLists.txt。
3. 根目录通过 src/CMakeLists.txt 的 add_subdirectory 注册模块。
4. 新改动中避免硬编码 x86_64 的头文件或库路径。
5. TensorRT 与 CUDA 路径优先通过可配置的 cache 变量管理。
6. 模块目标保持隔离（每个模块单独拥有自己的可执行或库目标）。
7. 依赖管理遵循 docs/dependency-management.md 中的约定

## 推荐模块结构
新增模块时，优先采用以下结构：

- <module>/CMakeLists.txt
- <module>/main.cpp（主入口）
- <module>/tensorrt_module.cpp + tensorrt_module.hpp（TensorRT 推理逻辑封装，推荐）
- <module>/export_<module>_onnx.py（ONNX 模型导出脚本）
- <module>/models/（可选，仅放该模块私有的小型样例资源）
- <module>/images/（可选，测试图片）

根目录模型目录策略：

- models/（全局共享，优先放跨模块复用资源、下载脚本与说明）

## Python 环境与导出脚本约定
- 使用 uv 管理 Python 虚拟环境，配置见 pyproject.toml 与 docs/python-environment.md。
- 在 Jetson 上安装 PyTorch 时，必须使用 NVIDIA 官方 JetPack 匹配的 wheel，禁止从 PyPI 盲装。
- 各模块的 ONNX 导出脚本应在模块目录内，命名格式统一为 export_<model>_onnx.py。
- 导出脚本产生的 .onnx 文件应放在模块的 models/ 子目录下，且不提交大体积模型文件到 Git。

## 平台与依赖常见陷阱
- 在 Jetson 上不要假设存在 /usr/include/x86_64-linux-gnu。
- Jetson 上 TensorRT 库通常位于 /usr/lib/aarch64-linux-gnu。
- CUDA 的 include/lib 路径应保持可配置，避免环境绑定。
- 避免将大体积模型二进制直接提交到仓库；优先保存下载脚本与校验说明。


## 代理改动策略
当任务要求优化 CMake 管理或新增模块时：
1. 先阅读根 CMakeLists.txt、src/CMakeLists.txt、现有模块 CMake 文件，以及 docs/dependency-management.md。
2. 涉及 Python 时，阅读 docs/python-environment.md 与 pyproject.toml。
3. 采用最小、渐进式重构。
4. 在保持现有行为的前提下，增强 add_subdirectory 式扩展能力。
5. 通过配置与构建命令验证改动。
6. 明确报告修改了哪些文件，以及原因。

## 范围边界
除非用户明确要求，否则不要在“自定义指导文件”任务中实现模型推理代码。此类任务仅聚焦构建结构与代理生产力提升。
