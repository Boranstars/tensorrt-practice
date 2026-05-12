#include "tensorrt_module.hpp"
#include "utils.hpp"

#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <thread>
#include <atomic>
#include <array>

#include <cuda_runtime_api.h>

#include <fmt/core.h>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

static constexpr std::string_view YOLOV5_ONNX_MODEL_PATH = "/home/jetson/Programs/tensorrt/tensorrt-practice/src/yolov5/models/yolov5s_nms.onnx";
static constexpr std::string_view COCO_NAMES_PATH = "/home/jetson/Programs/tensorrt/tensorrt-practice/src/yolov5/models/coco.names";

namespace {

bool isInteger(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') {
        i = 1;
    }
    if (i == s.size()) {
        return false;
    }
    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fmt::print("Usage: {} <video_path|camera_index> [--headless]\n", argv[0]);
        return -1;
    }

    std::string source = argv[1];
    bool headless = false;
    constexpr int kFpsPrintInterval = 30;
    constexpr size_t kMaxQueueSize = 2;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        }
    }

    auto logger = std::make_unique<Logger>(nvinfer1::ILogger::Severity::kINFO);
    TRTParams params{
        .onnxFilePath = YOLOV5_ONNX_MODEL_PATH.data(),
        .useDLA = false,
        .useFP16 = true,
        .batchSize = 1,
        .channels = 3,
        .input_h = 640,
        .input_w = 640,
        .output_size = 300 * 6,
    };

    TensorRTModule trt(std::move(logger), params);
    trt.initialize();

    const size_t inputElems = static_cast<size_t>(params.batchSize) *
                              static_cast<size_t>(params.channels) *
                              static_cast<size_t>(params.input_h) *
                              static_cast<size_t>(params.input_w);

    const auto classNames = loadClassNames(std::string(COCO_NAMES_PATH));

    cv::VideoCapture cap;
    if (isInteger(source)) {
        cap.open(std::stoi(source));
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        fmt::print("Error: failed to open video source: {}\n", source);
        return -1;
    }

    if (!headless) {
        cv::namedWindow("yolov5_videotest", cv::WINDOW_NORMAL);
    }

    std::queue<cv::Mat> frameQueue;
    std::mutex frameMutex;
    std::condition_variable frameCv;

    std::queue<cv::Mat> displayQueue;
    std::mutex displayMutex;
    std::condition_variable displayCv;

    std::atomic<bool> stopRequested{false};

    std::thread captureThread([&]() {
        cv::Mat frame;
        while (!stopRequested.load()) {
            if (!cap.read(frame)) {
                break;
            }
            if (frame.empty()) {
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                if (frameQueue.size() >= kMaxQueueSize) {
                    continue;
                }
                frameQueue.push(frame.clone());
            }
            frameCv.notify_one();
        }

        {
            std::lock_guard<std::mutex> lock(frameMutex);
            frameQueue.push(cv::Mat());
        }
        frameCv.notify_one();
    });

    std::thread inferThread([&]() {
        int frameCount = 0;
        cv::Mat shared_canvas(params.input_h, params.input_w, CV_8UC3);
        LetterboxResult box_info;
        std::vector<Detection> finalDets;
        finalDets.reserve(300);

        while (!stopRequested.load()) {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCv.wait(lock, [&]() {
                    return !frameQueue.empty() || stopRequested.load();
                });

                if (stopRequested.load() && frameQueue.empty()) {
                    break;
                }

                frame = frameQueue.front();
                frameQueue.pop();
            }

            if (frame.empty()) {
                break;
            }

            const auto t_cap = std::chrono::steady_clock::now();

            auto t0 = std::chrono::steady_clock::now();
            letterbox(frame, shared_canvas, box_info);
            cv::Mat rgbFloat;
            cv::cvtColor(shared_canvas, rgbFloat, cv::COLOR_BGR2RGB);
            rgbFloat.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

            float* hostInput = trt.getHostInput();
            const size_t hw = static_cast<size_t>(params.input_h) * params.input_w;
            std::array<cv::Mat, 3> chw{
                cv::Mat(params.input_h, params.input_w, CV_32FC1, hostInput + 0 * hw),
                cv::Mat(params.input_h, params.input_w, CV_32FC1, hostInput + 1 * hw),
                cv::Mat(params.input_h, params.input_w, CV_32FC1, hostInput + 2 * hw),
            };
            cv::split(rgbFloat, chw);
            const auto t_pre = std::chrono::steady_clock::now();

            auto inferStart = std::chrono::steady_clock::now();
            if (!trt.inferFromHostBuffer()) {
                fmt::print("Error: TensorRT inference failed on current frame.\n");
                continue;
            }
            auto inferEnd = std::chrono::steady_clock::now();
            const auto t_infer = inferEnd;

            postprocessYoloNmsOutput(trt.getHostOutput(),
                                     frame.cols,
                                     frame.rows,
                                     box_info.r,
                                     box_info.dw,
                                     box_info.dh,
                                     0.25F,
                                     finalDets);
            const auto t_post = std::chrono::steady_clock::now();
            drawDetections(frame, finalDets, classNames);

            auto t1 = std::chrono::steady_clock::now();
            const double e2eMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            const double inferMs =
                std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
            const double e2eFps = e2eMs > 0.0 ? 1000.0 / e2eMs : 0.0;
            const double inferFps = inferMs > 0.0 ? 1000.0 / inferMs : 0.0;

            if (!headless) {
                const std::string e2eText = cv::format("E2E FPS: %.2f", e2eFps);
                const std::string inferText = cv::format("Infer FPS: %.2f", inferFps);
                cv::putText(frame, e2eText, cv::Point(20, 35),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255),
                            2, cv::LINE_AA);
                cv::putText(frame, inferText, cv::Point(20, 70),
                            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0),
                            2, cv::LINE_AA);
            }

            ++frameCount;
            if (frameCount % kFpsPrintInterval == 0) {
                const double preMs = std::chrono::duration<double, std::milli>(t_pre - t_cap).count();
                const double inferStageMs = std::chrono::duration<double, std::milli>(t_infer - t_pre).count();
                const double postMs = std::chrono::duration<double, std::milli>(t_post - t_infer).count();
                fmt::print("[{}] Infer FPS: {:.2f}, E2E FPS: {:.2f}, detections: {}\n",
                           frameCount, inferFps, e2eFps, finalDets.size());
                fmt::print("    Stage Latency | Pre-process: {:.2f} ms, Inference: {:.2f} ms, Post-process: {:.2f} ms\n",
                           preMs, inferStageMs, postMs);
            }

            if (!headless) {
                std::lock_guard<std::mutex> lock(displayMutex);
                if (displayQueue.size() >= kMaxQueueSize) {
                    displayQueue.pop();
                }
                displayQueue.push(frame);
                displayCv.notify_one();
            }
        }

        if (!headless) {
            std::lock_guard<std::mutex> lock(displayMutex);
            displayQueue.push(cv::Mat());
            displayCv.notify_one();
        }
    });

    if (!headless) {
        while (!stopRequested.load()) {
            cv::Mat rendered;
            {
                std::unique_lock<std::mutex> lock(displayMutex);
                displayCv.wait(lock, [&]() {
                    return !displayQueue.empty() || stopRequested.load();
                });

                if (displayQueue.empty()) {
                    continue;
                }
                rendered = displayQueue.front();
                displayQueue.pop();
            }

            if (rendered.empty()) {
                break;
            }

            cv::imshow("yolov5_videotest", rendered);
            const int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                stopRequested.store(true);
                frameCv.notify_all();
                displayCv.notify_all();
                break;
            }
        }

        stopRequested.store(true);
        frameCv.notify_all();
        displayCv.notify_all();
    }

    if (captureThread.joinable()) {
        captureThread.join();
    }
    if (inferThread.joinable()) {
        inferThread.join();
    }

    cap.release();
    if (!headless) {
        cv::destroyAllWindows();
    }

    return 0;
}
