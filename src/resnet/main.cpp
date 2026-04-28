#include "tensorrt_module.hpp"

#include <fmt/core.h>
#include <string_view>

static constexpr std::string_view RESNET_ONNX_PATH =
    "/home/jetson/Programs/tensorrt/tensorrt-practice/src/resnet/models/"
    "resnet.onnx";

int main(int argc, char** argv) {
    fmt::print("resnet module scaffold is ready.\n");
    if (argc < 2) {
        fmt::print(stderr, "Usage: {} <image_path>\n", argv[0]);
        return 1;
    }

    TRTParams params{
        .useDLA = false,
        .useFP16 = true,
        .batchSize = 1,
        .channels = 3,
        .input_h = 224,
        .input_w = 224,
        .output_size = 1000,
  
        
    };
    params.onnxFilePath = RESNET_ONNX_PATH.data();

    TensorRTModule module(std::make_unique<Logger>(nvinfer1::ILogger::Severity::kINFO),
                          params);
    module.initialize();

    // 单次推理 + 打印结果
    module.infer(argv[1]);

    // Benchmark: 1000 次推理
    module.benchmark(1000);
    return 0;
}
