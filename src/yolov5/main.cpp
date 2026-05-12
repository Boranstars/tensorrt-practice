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
        .batchSize = 1,
        .channels = 3,
        .input_h = 640,
        .input_w = 640,
        .output_size = 84 * 8400, // 84 classes * 8400 boxes
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

    const auto classNames = loadClassNames(std::string(COCO_NAMES_PATH));

    cv::Mat sharedCanvas(params.input_h, params.input_w, CV_8UC3);
    LetterboxResult letterboxResult;
    letterbox(img, sharedCanvas, letterboxResult);
    // 使用dnn模块的blobFromImage函数进行预处理，得到CHW格式的输入数据
    cv::Mat inputBlob = cv::dnn::blobFromImage(sharedCanvas, 1.0 / 255.0, cv::Size(640,640), cv::Scalar(), true, false);
    if (!trt.infer(inputBlob.ptr<float>(), static_cast<size_t>(inputBlob.total()))) {
        fmt::print("Error: TensorRT inference failed.\n");
        return -1;
    }
    const float* output = trt.getHostOutput();

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    std::vector<Detection> detections;
    bboxes.reserve(8400);
    scores.reserve(8400);
    classIds.reserve(8400);
    detections.reserve(8400);
    postprocessYolov5su(output, img.cols, img.rows,
                        letterboxResult.r,
                        letterboxResult.dw,
                        letterboxResult.dh,
                        0.25F,
                        0.45F,
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
