---
name: add-trt-module
description: 为 TensorRT 学习工程新增一个可维护的子模块，并按约定接入 CMake（仅脚手架与构建管理，不实现推理逻辑）
---

## 适用场景
当用户希望新增学习模块（例如 mlp、yolo、api-experiments）并统一管理 CMake 时，使用本技能。

## 输入
- 模块名（必填）：例如 `mlp`、`yolo`、`api_experiments`
- 可选项：是否需要模块私有 models 目录

## 约束
1. 仅创建或更新模块脚手架与 CMake 文件，不实现具体推理代码。
2. 顶层 CMakeLists.txt 只保留共享配置，模块逻辑放到子目录 CMakeLists.txt。
3. 不在新改动中硬编码 x86_64 路径。
4. TensorRT/CUDA 路径优先使用可配置 cache 变量。
5. 在 Jetson 场景下兼容 aarch64 常见路径（如 /usr/lib/aarch64-linux-gnu）。
6. 子模块代码默认扁平化组织，不强制拆分 src/include。
7. 采用双层 models 策略：根目录 models 用于共享资源，模块内 models 仅用于模块私有小样例。
8. 所有模块目录统一放在仓库 `src/` 下。

## 目标目录结构
为 `src/<module>` 创建：
- `src/<module>/CMakeLists.txt`
- `src/<module>/*.cpp` 或 `src/<module>/*.cu`（扁平化）
- `src/<module>/*.h`（可选）
- `src/<module>/models/`（可选，模块私有小样例）

在 `src/` 下维护模块注册文件：
- `src/CMakeLists.txt`（通过 add_subdirectory 注册各模块）

在根目录按需维护：
- `models/`（全局共享模型资源、下载脚本与说明）

## 执行步骤
1. 读取根 CMakeLists.txt，确认是否已有模块注册区。
2. 确保存在 `src/CMakeLists.txt`，并由根 CMakeLists.txt 通过 `add_subdirectory(src)` 引入。
3. 在 `src/<module>` 下创建模块目录与最小脚手架文件（必要时创建占位源文件），默认采用模块内扁平化布局。
4. 在 `src/CMakeLists.txt` 中通过 `add_subdirectory(<module>)` 注册模块。
5. 在 `src/<module>/CMakeLists.txt` 定义模块目标，并只在模块内声明其包含目录/链接依赖。
6. 按双层 models 策略处理资源：共享资源放根 `models/`，模块私有样例放 `src/<module>/models/`。
7. 运行配置与构建验证：
   - `cmake -S . -B build -G Ninja`
   - `cmake --build build`
8. 汇总输出：
   - 新增/修改文件列表
   - 每个文件的作用
   - 是否通过配置与构建

## 输出格式要求
- 先给结果摘要（是否完成 + 是否构建通过）。
- 然后给文件变更表（文件、动作、原因）。
- 最后给下一步建议（如是否补充示例 main.cpp 或模型资源说明）。

## 失败处理
- 若构建失败，优先报告可定位的 CMake 问题与最小修复建议。
- 不得为“通过构建”而引入平台绑定硬编码路径。
