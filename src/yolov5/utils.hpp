#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct LetterboxResult {
    cv::Mat image;
    float r{1.0F};
    int dw{0};
    int dh{0};
};

struct Detection {
    cv::Rect box;
    int classId{-1};
    float score{0.0F};
};

LetterboxResult letterbox(const cv::Mat& src,
                          int targetW = 640,
                          int targetH = 640,
                          const cv::Scalar& padColor = cv::Scalar(114, 114, 114));

std::vector<Detection> postprocessYolov5su(
    const float* output,
    int originalW,
    int originalH,
    float r,
    int dw,
    int dh,
    float scoreThreshold = 0.25F,
    float nmsThreshold = 0.45F);

std::vector<std::string> loadClassNames(const std::string& filePath);

void drawDetections(cv::Mat& image,
                    const std::vector<Detection>& detections,
                    const std::vector<std::string>& classNames);
