#include "tensorrt_module.hpp"

#include <fmt/core.h>
#include <string_view>

static constexpr std::string_view GOOGLENET_ONNX_PATH =
    "/home/jetson/Programs/tensorrt/tensorrt-practice/src/googlenet/models/"
    "googlenet.onnx";

int main() {
    fmt::print("googlenet module scaffold is ready.\n");

    TRTParams params;
    params.onnxFilePath = GOOGLENET_ONNX_PATH.data();

    TensorRTModule module(std::make_unique<Logger>(nvinfer1::ILogger::Severity::kINFO),
                          params);
    module.initialize();
    module.infer();
    return 0;
}
