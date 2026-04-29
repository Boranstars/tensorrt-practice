#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

LetterboxResult letterbox(const cv::Mat& src,
                          int targetW,
                          int targetH,
                          const cv::Scalar& padColor) {
    LetterboxResult result;
    if (src.empty() || targetW <= 0 || targetH <= 0) {
        return result;
    }

    const int oriW = src.cols;
    const int oriH = src.rows;
    result.r = std::min(static_cast<float>(targetW) / static_cast<float>(oriW),
                        static_cast<float>(targetH) / static_cast<float>(oriH));

    const int resizedW = static_cast<int>(std::round(static_cast<float>(oriW) * result.r));
    const int resizedH = static_cast<int>(std::round(static_cast<float>(oriH) * result.r));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resizedW, resizedH), 0.0, 0.0, cv::INTER_LINEAR);

    const int padW = targetW - resizedW;
    const int padH = targetH - resizedH;

    const int left = padW / 2;
    const int right = padW - left;
    const int top = padH / 2;
    const int bottom = padH - top;

    result.dw = left;
    result.dh = top;

    cv::copyMakeBorder(resized, result.image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, padColor);
    return result;
}

std::vector<Detection> postprocessYolov5su(
    const float* output,
    int originalW,
    int originalH,
    float r,
    int dw,
    int dh,
    float scoreThreshold,
    float nmsThreshold) {
    static constexpr int kNumAttrs = 84;
    static constexpr int kNumCandidates = 8400;

    std::vector<Detection> finalDetections;
    if (output == nullptr || originalW <= 0 || originalH <= 0 || r <= 0.0F) {
        return finalDetections;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    boxes.reserve(kNumCandidates);
    scores.reserve(kNumCandidates);
    classIds.reserve(kNumCandidates);

    for (int i = 0; i < kNumCandidates; ++i) {
        const float cx = output[0 * kNumCandidates + i];
        const float cy = output[1 * kNumCandidates + i];
        const float w = output[2 * kNumCandidates + i];
        const float h = output[3 * kNumCandidates + i];

        int bestClass = -1;
        float bestScore = -std::numeric_limits<float>::infinity();
        for (int attr = 4; attr < kNumAttrs; ++attr) {
            const float s = output[attr * kNumCandidates + i];
            if (s > bestScore) {
                bestScore = s;
                bestClass = attr - 4;
            }
        }

        if (bestScore <= scoreThreshold) {
            continue;
        }

        const float x1 = (cx - 0.5F * w - static_cast<float>(dw)) / r;
        const float y1 = (cy - 0.5F * h - static_cast<float>(dh)) / r;
        const float x2 = (cx + 0.5F * w - static_cast<float>(dw)) / r;
        const float y2 = (cy + 0.5F * h - static_cast<float>(dh)) / r;

        const int left = std::max(0, static_cast<int>(std::round(x1)));
        const int top = std::max(0, static_cast<int>(std::round(y1)));
        const int right = std::min(originalW, static_cast<int>(std::round(x2)));
        const int bottom = std::min(originalH, static_cast<int>(std::round(y2)));

        const int boxW = right - left;
        const int boxH = bottom - top;
        if (boxW <= 0 || boxH <= 0) {
            continue;
        }

        boxes.emplace_back(left, top, boxW, boxH);
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<int> kept;
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, kept);

    finalDetections.reserve(kept.size());
    for (int idx : kept) {
        finalDetections.push_back(Detection{boxes[idx], classIds[idx], scores[idx]});
    }
    return finalDetections;
}

std::vector<std::string> loadClassNames(const std::string& filePath) {
    std::vector<std::string> names;
    std::ifstream ifs(filePath);
    if (!ifs.is_open()) {
        return names;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}

void drawDetections(cv::Mat& image,
                    const std::vector<Detection>& detections,
                    const std::vector<std::string>& classNames) {
    for (const auto& det : detections) {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        std::string classText = cv::format("id:%d", det.classId);
        if (det.classId >= 0 && det.classId < static_cast<int>(classNames.size())) {
            classText = classNames[det.classId];
        }
        const std::string label = cv::format("%s %.2f", classText.c_str(), det.score);
        int baseline = 0;
        const cv::Size textSize =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int textX = det.box.x;
        const int textY = std::max(0, det.box.y - 4);
        const cv::Rect bgRect(textX,
                              std::max(0, textY - textSize.height - baseline),
                              textSize.width + 4,
                              textSize.height + baseline + 4);
        cv::rectangle(image, bgRect, cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(textX + 2, textY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1,
                    cv::LINE_AA);
    }
}
