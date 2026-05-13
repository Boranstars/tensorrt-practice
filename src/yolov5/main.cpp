#include "tensorrt_module.hpp"
#include "utils.hpp"

#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <fmt/core.h>
#include <memory>

static constexpr std::string_view YOLOV5_ONNX_MODEL_PATH = "/home/jetson/Programs/tensorrt/tensorrt-practice/src/yolov5/models/yolov5su.onnx";
static constexpr std::string_view COCO_NAMES_PATH = "/home/jetson/Programs/tensorrt/tensorrt-practice/src/yolov5/models/coco.names";

namespace fs = std::filesystem;




int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print("Usage: {} <image_path>\n", argv[0]);
        return -1;
    }

    const std::string imagePath = argv[1];
    auto logger = std::make_unique<Logger>(Severity::kINFO);

    TRTParams params {
        .onnxFilePath = YOLOV5_ONNX_MODEL_PATH.data(),
        .useDLA = false,
        .useFP16 = true,
        // .batchSize = 1,
        // .channels = 3,
        // .input_h = 640,
        // .input_w = 640,
        // .output_size = 84 * 8400, // 84 classes * 8400 boxes
    };
    params.onnxFilePath = YOLOV5_ONNX_MODEL_PATH.data();

    TensorRTModule trt(std::move(logger), params);
    trt.initialize();   
    fmt::print("yolov5_demo scaffold is ready.\n");

    // 推理：
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        fmt::print("Error: Could not read image file: {}\n", imagePath);
        return -1;
    }

    const auto& inputMeta = trt.getInputTensorMetadata();
    const int input_h = inputMeta.dims.d[2];
    const int input_w = inputMeta.dims.d[3];

    const auto classNames = loadClassNames(std::string(COCO_NAMES_PATH));

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

    int num_attributes = outdims.d[1]; // 每个候选框的属性数量（4个坐标 + 类别分数）
    int num_boxes = outdims.d[2]; // 候选框数量
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    std::vector<Detection> detections;
    bboxes.reserve(num_boxes);
    scores.reserve(num_boxes);
    classIds.reserve(num_boxes);
    detections.reserve(num_boxes);

    PostProcessConfig config;
    config.num_classes = num_attributes - 4; // 类别分数数量 = 总属性数量 - 4（坐标）
    config.num_boxes = num_boxes;
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
