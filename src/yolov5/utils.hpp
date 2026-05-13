#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct LetterboxResult {
    float r{1.0F};
    int dw{0};
    int dh{0};
};

struct Detection {
    cv::Rect box;
    int classId{-1};
    float score{0.0F};
};

struct PostProcessConfig {
    int num_classes;  // 动态获取：比如 80
    int num_boxes;    // 动态获取：比如 8400
    int originalW;
    int originalH;
    float r;
    int dw;
    int dh;
    float scoreThreshold = 0.25f;
    float nmsThreshold = 0.45f;
};

void letterbox(const cv::Mat& src,
               cv::Mat& canvas,
               LetterboxResult& result,
               const cv::Scalar& padColor = cv::Scalar(114, 114, 114));

void postprocessYolov5su(
    const float* output,
    const PostProcessConfig& config,
    std::vector<cv::Rect>& boxes,
    std::vector<float>& scores,
    std::vector<int>& classIds,
    std::vector<Detection>& finalDetections);

std::vector<std::string> loadClassNames(const std::string& filePath);

void drawDetections(cv::Mat& image,
                    const std::vector<Detection>& detections,
                    const std::vector<std::string>& classNames);
