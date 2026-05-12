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

void letterbox(const cv::Mat& src,
               cv::Mat& canvas,
               LetterboxResult& result,
               const cv::Scalar& padColor = cv::Scalar(114, 114, 114));

void postprocessYolov5su(
    const float* output,
    int originalW,
    int originalH,
    float r,
    int dw,
    int dh,
    float scoreThreshold,
    float nmsThreshold,
    std::vector<cv::Rect>& boxes,
    std::vector<float>& scores,
    std::vector<int>& classIds,
    std::vector<Detection>& finalDetections);

std::vector<std::string> loadClassNames(const std::string& filePath);

void drawDetections(cv::Mat& image,
                    const std::vector<Detection>& detections,
                    const std::vector<std::string>& classNames);
