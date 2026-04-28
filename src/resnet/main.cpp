#include "tensorrt_module.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fmt/core.h>
#include <stdexcept>
#include <vector>
#include <string_view>

static constexpr std::string_view RESNET_ONNX_PATH =
    "/home/jetson/Programs/tensorrt/tensorrt-practice/src/resnet/models/"
    "resnet.onnx";

static std::vector<float> loadImageAsCHWFloat(const std::string& imagePath,
                                              int width,
                                              int height) {
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("failed to read image: " + imagePath);
    }

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(width, height));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    // ImageNet 标准均值和标准差
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_val[3] = {0.229f, 0.224f, 0.225f};

    std::vector<float> chw(3 * width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec3f& pixel = f32.at<cv::Vec3f>(y, x);
            
            // HWC 转 CHW 的同时，减去均值，除以标准差
            chw[0 * width * height + y * width + x] = (pixel[0] - mean[0]) / std_val[0]; // R
            chw[1 * width * height + y * width + x] = (pixel[1] - mean[1]) / std_val[1]; // G
            chw[2 * width * height + y * width + x] = (pixel[2] - mean[2]) / std_val[2]; // B
        }
    }
    return chw;
}

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

    auto input = loadImageAsCHWFloat(argv[1], params.input_w, params.input_h);

    // 单次推理 + 打印结果
    module.infer(input.data());

    // Benchmark: 1000 次推理
    module.benchmark(1000, input.data());
    return 0;
}
