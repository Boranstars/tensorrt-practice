# tensorrt-practice 代理指南

## 目标
本工作区用于学习 TensorRT 推理，包含多个子模块，例如 mlp、yolo，以及面向特定 API 的实验。

代理在执行任务时，应优先保证 CMake 组织可维护、可扩展，便于持续新增学习模块。

## 当前项目事实
- 构建系统：CMake + Ninja（build 目录中已存在 Ninja 构建文件）。
- 根构建文件：CMakeLists.txt。
- 已有学习模块目录：mlp（当前为空）。
- 推荐新增共享模型目录：models/（用于跨模块共享资源或下载脚本）。
- 目标环境：Linux，常见为 Jetson aarch64。

## 构建命令
除非用户另有说明，默认使用：

- 配置：cmake -S . -B build -G Ninja
- 构建：cmake --build build

如果做了结构调整后出现 CMake 缓存不一致问题，清理 build 并重新配置。

## CMake 组织规则
1. 顶层 CMakeLists.txt 只保留共享项目配置。
2. 每个学习主题使用独立目录，并在该目录内维护自己的 CMakeLists.txt。
3. 根目录通过 add_subdirectory 注册模块。
4. 新改动中避免硬编码 x86_64 的头文件或库路径。
5. TensorRT 与 CUDA 路径优先通过可配置的 cache 变量管理。
6. 模块目标保持隔离（每个模块单独拥有自己的可执行或库目标）。

## 推荐模块结构
新增模块时，优先采用以下结构：

- <module>/CMakeLists.txt
- <module>/*.cpp 或 <module>/*.cu（模块内扁平化组织）
- <module>/*.h（可选）
- <module>/models/（可选，仅放该模块私有的小型样例资源）

根目录模型目录策略：

- models/（全局共享，优先放跨模块复用资源、下载脚本与说明）

## 平台与依赖常见陷阱
- 在 Jetson 上不要假设存在 /usr/include/x86_64-linux-gnu。
- Jetson 上 TensorRT 库通常位于 /usr/lib/aarch64-linux-gnu。
- CUDA 的 include/lib 路径应保持可配置，避免环境绑定。
- 避免将大体积模型二进制直接提交到仓库；优先保存下载脚本与校验说明。

## 代理改动策略
当任务要求优化 CMake 管理时：
1. 先阅读根 CMakeLists.txt 与现有模块 CMake 文件。
2. 采用最小、渐进式重构。
3. 在保持现有行为的前提下，增强 add_subdirectory 式扩展能力。
4. 通过配置与构建命令验证改动。
5. 明确报告修改了哪些文件，以及原因。

## 范围边界
除非用户明确要求，否则不要在“自定义指导文件”任务中实现模型推理代码。此类任务仅聚焦构建结构与代理生产力提升。
