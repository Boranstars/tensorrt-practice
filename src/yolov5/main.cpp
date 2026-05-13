#include "tensorrt_module.hpp"
#include "utils.hpp"

#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <fmt/core.h>
#include <array>
#include <memory>

static constexpr std::string_view YOLOV8_ONNX_MODEL_PATH = "/home/jetson/Programs/tensorrt/tensorrt-practice/src/yolov5/models/best.onnx";
static const std::array<std::string, 14> AUTOAIM_CLASS_NAMES{
    "B1", "B2", "B3", "B4", "B5", "BO", "BS",
    "R1", "R2", "R3", "R4", "R5", "RO", "RS",
};

namespace fs = std::filesystem;




int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print("Usage: {} <image_path>\n", argv[0]);
        return -1;
    }

    const std::string imagePath = argv[1];
    auto logger = std::make_unique<Logger>(Severity::kINFO);

    TRTParams params {
        .onnxFilePath = YOLOV8_ONNX_MODEL_PATH.data(),
        .useDLA = false,
        .useFP16 = true,
    };

    TensorRTModule trt(std::move(logger), params);
    trt.initialize();   
    fmt::print("yolov8 autoaim demo scaffold is ready.\n");

    // 推理：
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        fmt::print("Error: Could not read image file: {}\n", imagePath);
        return -1;
    }

    const auto& inputMeta = trt.getInputTensorMetadata();
    const int input_h = inputMeta.dims.d[2];
    const int input_w = inputMeta.dims.d[3];

    const std::vector<std::string> classNames(AUTOAIM_CLASS_NAMES.begin(),
                                              AUTOAIM_CLASS_NAMES.end());

    cv::Mat sharedCanvas(input_h, input_w, CV_8UC3);
    LetterboxResult letterboxResult;
    letterbox(img, sharedCanvas, letterboxResult);
    // 使用dnn模块的blobFromImage函数进行预处理，得到CHW格式的输入数据
    cv::Mat inputBlob = cv::dnn::blobFromImage(sharedCanvas, 1.0 / 255.0, cv::Size(input_w, input_h), cv::Scalar(), true, false);
    if (!trt.infer(inputBlob.ptr<float>(), static_cast<size_t>(inputBlob.total()))) {
        fmt::print("Error: TensorRT inference failed.\n");
        return -1;
    }


    // 后处理
    const float* output = trt.getHostOutput();
    auto outoputMetadata = trt.getOutputTensorMetadata();
    const auto outdims = outoputMetadata.dims;
    fmt::print("Output tensor dims: nbDims: {}, dims: ", outdims.nbDims);
    for (int i = 0; i < outdims.nbDims; ++i) {
        fmt::print("{} ", outdims.d[i]);
    }
    fmt::print("\n");

    constexpr int kAutoaimNumClasses = static_cast<int>(AUTOAIM_CLASS_NAMES.size());
    constexpr int kKeypointAttrs = 3;
    const int num_attributes = outdims.d[1];
    const int num_boxes = outdims.d[2];
    const int num_keypoints = (num_attributes - 4 - kAutoaimNumClasses) / kKeypointAttrs;
    if (num_attributes != 30 || num_keypoints != 4) {
        fmt::print("Warning: unexpected autoaim output layout, attrs={}, keypoints={}\n",
                   num_attributes, num_keypoints);
    }
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    std::vector<Detection> detections;
    bboxes.reserve(num_boxes);
    scores.reserve(num_boxes);
    classIds.reserve(num_boxes);
    detections.reserve(num_boxes);

    PostProcessConfig config;
    config.num_classes = kAutoaimNumClasses;
    config.num_boxes = num_boxes;
    config.num_keypoints = num_keypoints;
    config.originalW = img.cols;
    config.originalH = img.rows;    
    config.r = letterboxResult.r;
    config.dw = letterboxResult.dw;
    config.dh = letterboxResult.dh;
    config.scoreThreshold = 0.25F;
    config.nmsThreshold = 0.45F;

    postprocessYolov5su(output, config,
                        bboxes,
                        scores,
                        classIds,
                        detections);

    drawDetections(img, detections, classNames);

    const fs::path inputPath(imagePath);
    const fs::path outputPath =
        inputPath.parent_path() /
        fs::path(inputPath.stem().string() + "_result" + inputPath.extension().string());
    if (!cv::imwrite(outputPath, img)) {
        fmt::print("Error: Failed to save result image to {}\n", outputPath.string());
        return -1;
    }
    fmt::print("Detections: {}\n", detections.size());
    fmt::print("Saved result image: {}\n", outputPath.string());
    

    trt.benchmark(1000, inputBlob.ptr<float>());
    return 0;
}
