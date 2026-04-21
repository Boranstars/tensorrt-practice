# CMake 依赖管理约定（全局依赖 + 模块私有依赖）

本文档用于说明在本仓库中如何管理第三方依赖，兼顾可维护性与模块隔离。

## 目标
- 根 CMake 只维护共享配置与共享依赖。
- 子模块只声明自己需要的依赖。
- 避免全局 include_directories/link_directories 污染所有目标。

## 1. 什么时候放全局依赖
适用于这些场景：
- 多个模块都会用到的库（例如 fmt、spdlog）。
- 工具性质的公共库（日志、格式化、CLI 参数）。

推荐做法：
1. 在根 CMakeLists 中 find_package 或 FetchContent。
2. 用 INTERFACE 目标统一封装共享依赖。
3. 子模块通过 target_link_libraries(... PRIVATE 共享目标) 引用。

示例（以 fmt 为例，根 CMakeLists）：

```cmake
# 方案 A：系统已安装 fmt（推荐优先）
find_package(fmt CONFIG QUIET)

# 方案 B：未安装时自动拉取
if(NOT fmt_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.0.2
    )
    FetchContent_MakeAvailable(fmt)
endif()

# 共享公共依赖聚合目标
add_library(trt_practice_common INTERFACE)

target_link_libraries(trt_practice_common INTERFACE
    trt_practice_deps
    fmt::fmt
)
```

子模块中使用：

```cmake
target_link_libraries(mlp_demo
    PRIVATE trt_practice_common
)
```

## 2. 什么时候放模块私有依赖
适用于这些场景：
- 仅某一个模块需要（例如 yolo 模块用 OpenCV，mlp 模块不需要）。
- 实验性质依赖，不希望影响所有模块构建。

推荐做法：
1. 在模块自己的 CMakeLists 里 find_package。
2. 只给该模块目标链接，不上升到根 CMake。
3. 可以给该模块单独加 BUILD 开关，减少无关模块构建失败。

示例（yolo/CMakeLists，OpenCV 私有依赖）：

```cmake
add_executable(yolo_demo
    main.cpp
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

target_link_libraries(yolo_demo
    PRIVATE
        trt_practice_deps
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
)
```

可选：根 CMake 增加模块开关

```cmake
option(BUILD_YOLO "Build yolo module" ON)
if(BUILD_YOLO)
    add_subdirectory(yolo)
endif()
```

## 3. 推荐决策规则
- 依赖被 2 个及以上模块使用：优先考虑全局共享依赖。
- 依赖仅单模块使用：放模块私有依赖。
- 依赖体积大、环境差异大（如 OpenCV、TensorFlow）：优先模块私有，避免牵连全局。

## 4. 与模型目录策略的配合
- 根 models/：跨模块共享模型、下载脚本、校验说明。
- 模块内 models/：仅该模块私有小样例资源。
- 不建议提交大体积模型二进制文件到仓库。

## 5. 常见坑
- 不要在根 CMake 对所有目标强行链接 OpenCV 等重依赖。
- 不要把模块私有依赖写进 trt_practice_deps。
- 不要硬编码 x86_64 路径，路径应通过 cache 变量可配置。

## 6. 一句话总结
- fmt 这类通用库：做成全局共享依赖。
- OpenCV 这类模块专属库：放到对应模块私有依赖。
