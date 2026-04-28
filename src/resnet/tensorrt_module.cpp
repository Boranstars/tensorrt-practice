#include "tensorrt_module.hpp"

#include <NvOnnxParser.h>
#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>

#define CHECK(status)                                                          \
    do {                                                                       \
        auto ret = (status);                                                   \
        if (ret != 0) {                                                        \
            fmt::print(stderr, "Cuda failure: {} at {}:{}\\n", ret, __FILE__,   \
                       __LINE__);                                              \
            abort();                                                           \
        }                                                                      \
    } while (0)

namespace fs = std::filesystem;
using namespace nvinfer1;

static void loadImageToPinnedMemory(const std::string &imagePath,
                                    float *targetBuffer,
                                    int width,
                                    int height) {
    if (targetBuffer == nullptr) {
        throw std::runtime_error("targetBuffer is null");
    }

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

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_val[3] = {0.229f, 0.224f, 0.225f};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec3f &pixel = f32.at<cv::Vec3f>(y, x);
            targetBuffer[0 * width * height + y * width + x] =
                (pixel[0] - mean[0]) / std_val[0];
            targetBuffer[1 * width * height + y * width + x] =
                (pixel[1] - mean[1]) / std_val[1];
            targetBuffer[2 * width * height + y * width + x] =
                (pixel[2] - mean[2]) / std_val[2];
        }
    }
}

TensorRTModule::TensorRTModule(std::unique_ptr<nvinfer1::ILogger> logger,
                               const TRTParams &config)
    : m_logger(std::move(logger)), m_params(config) {}

TensorRTModule::~TensorRTModule() {
    if (deviceInput)
        cudaFree(deviceInput);
    if (deviceOutput)
        cudaFree(deviceOutput);
    if (m_pinnedInput)
        cudaFreeHost(m_pinnedInput);
    if (m_pinnedOutput)
        cudaFreeHost(m_pinnedOutput);
    if (m_stream)
        cudaStreamDestroy(m_stream);
}

void TensorRTModule::parserModel(const std::string &onnxFilePath,
                                 INetworkDefinition *network) {
    auto parser = nvonnxparser::createParser(*network, *m_logger);
    if (!parser->parseFromFile(onnxFilePath.c_str(),
                               static_cast<int>(Severity::kINFO))) {
        const auto msg = fmt::format("Failed to parse ONNX file: {}", onnxFilePath);
        m_logger->log(Severity::kERROR, msg.c_str());
        return;
    }
}

auto TensorRTModule::createEngine(std::unique_ptr<nvinfer1::IBuilder> builder,
                                  std::unique_ptr<nvinfer1::INetworkDefinition> network)
    -> std::unique_ptr<nvinfer1::ICudaEngine> {
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>{
        builder->createBuilderConfig()};

    if (m_params.useDLA) {
        if (builder->getNbDLACores() > 0) {
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
            config->setFlag(BuilderFlag::kFP16);
            config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            config->setDLACore(0);
            m_logger->log(Severity::kINFO, "Using DLA for inference.");
        } else {
            m_logger->log(Severity::kWARNING,
                          "DLA not available. Fallingback to GPU.");
        }
    }

    if(m_params.useFP16) {
        config->setFlag(BuilderFlag::kFP16);
        m_logger->log(Severity::kINFO, "Using FP16 precision for inference.");
    }

    size_t free{0}, total{0};
    CHECK(cudaMemGetInfo(&free, &total));
    m_logger->log(Severity::kINFO,
                  fmt::format("GPU memory - free: {} MB, total: {} MB",
                              free / (1024 * 1024), total / (1024 * 1024))
                      .c_str());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, free);

    auto output = network->getOutput(0);
    network->unmarkOutput(*output);
    m_logger->log(
        Severity::kINFO,
        fmt::format("Original output tensor name: {}", output->getName()).c_str());
    auto softmax = network->addSoftMax(*output);
    softmax->setAxes(1 << 1);
    softmax->getOutput(0)->setName("softmax_output");
    network->markOutput(*softmax->getOutput(0));

    m_logger->log(Severity::kINFO,
                  fmt::format("New output tensor name: {}",
                              softmax->getOutput(0)->getName())
                      .c_str());
    int nbDims = network->getOutput(0)->getDimensions().nbDims;
    m_logger->log(
        Severity::kINFO,
        fmt::format("Output tensor dimensions: nbDims: {}, dims: ", nbDims)
            .c_str());

    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>{
        builder->buildEngineWithConfig(*network, *config)};
    return engine;
}

void TensorRTModule::serializeEngine(std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                                     std::string_view engineFilePath) {
    auto serializedEngine =
        std::unique_ptr<nvinfer1::IHostMemory>{engine->serialize()};
    assert(serializedEngine);
    std::ofstream engineFile(std::string(engineFilePath), std::ios::binary);

    if (!engineFile) {
        m_logger->log(Severity::kERROR,
                      "Failed to open engine file for writing.");
        return;
    }
    engineFile.write(static_cast<const char *>(serializedEngine->data()),
                     serializedEngine->size());
    engineFile.close();

    m_logger->log(
        Severity::kINFO,
        fmt::format("Engine serialized to {}", engineFilePath).c_str());
}

void TensorRTModule::doInference() {
    CHECK(cudaMemcpyAsync(deviceInput, m_pinnedInput, m_inputSize,
                          cudaMemcpyHostToDevice, m_stream));
    if (!m_context->enqueueV3(m_stream))
        m_logger->log(Severity::kERROR, "Failed to execute inference.");
    CHECK(cudaMemcpyAsync(m_pinnedOutput, deviceOutput, m_outputSize,
                          cudaMemcpyDeviceToHost, m_stream));
    CHECK(cudaStreamSynchronize(m_stream));
}

void TensorRTModule::initialize(const TRTParams &buildConfig) {
    if (!buildConfig.onnxFilePath.empty()) {
        m_params = buildConfig;
    }

    if (m_params.onnxFilePath.empty()) {
        m_logger->log(Severity::kERROR, "onnxFilePath is empty in TRTParams.");
        return;
    }

    m_runtime.reset(nvinfer1::createInferRuntime(*m_logger));
    assert(m_runtime != nullptr);

    std::string engineFilePath =
        m_params.onnxFilePath.substr(0, m_params.onnxFilePath.find_last_of('.')) +
        ".engine";

    if (fs::exists(engineFilePath)) {
        m_logger->log(
            Severity::kINFO,
            fmt::format("Loading serialized engine: {}", engineFilePath).c_str());
        std::ifstream file(engineFilePath, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        m_engine.reset(m_runtime->deserializeCudaEngine(data.data(), data.size()));
    } else {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>{
            nvinfer1::createInferBuilder(*m_logger)};
        assert(builder != nullptr);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>{
            builder->createNetworkV2(0U)};
        assert(network != nullptr);
        parserModel(m_params.onnxFilePath, network.get());
        m_engine = createEngine(std::move(builder), std::move(network));
        serializeEngine(m_engine, engineFilePath);
    }
    assert(m_engine != nullptr);

    m_context.reset(m_engine->createExecutionContext());
    assert(m_context != nullptr);

    inputTensorName = m_engine->getIOTensorName(0);
    outputTensorName = m_engine->getIOTensorName(1);

        m_inputSize = static_cast<size_t>(m_params.batchSize) * m_params.channels *
                                    m_params.input_h * m_params.input_w * sizeof(float);
        m_outputSize = static_cast<size_t>(m_params.batchSize) * m_params.output_size *
                                     sizeof(float);

    CHECK(cudaMalloc(&deviceInput,
                                         m_inputSize));
    CHECK(cudaMalloc(&deviceOutput,
                                         m_outputSize));
        CHECK(cudaHostAlloc(reinterpret_cast<void **>(&m_pinnedInput), m_inputSize,
                                                cudaHostAllocDefault));
        CHECK(cudaHostAlloc(reinterpret_cast<void **>(&m_pinnedOutput), m_outputSize,
                                                cudaHostAllocDefault));
    CHECK(cudaStreamCreate(&m_stream));

    m_context->setTensorAddress(inputTensorName.c_str(), deviceInput);
    m_context->setTensorAddress(outputTensorName.c_str(), deviceOutput);

    m_logger->log(Severity::kINFO,
                  "TensorRTModule initialized. Resources allocated.");
}

void TensorRTModule::infer(const std::string &imagePath) {
    try {
        loadImageToPinnedMemory(imagePath, m_pinnedInput, m_params.input_w,
                                m_params.input_h);
    } catch (const std::exception &e) {
        m_logger->log(Severity::kERROR, e.what());
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    doInference();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start);

    for (int i = 0; i < m_params.batchSize * m_params.output_size; ++i)
        fmt::print("{:.6f} ", m_pinnedOutput[i]);
    fmt::print("\n");

    int maxClass = 0;
    float maxProb = m_pinnedOutput[0];
    for (int i = 1; i < m_params.output_size; ++i)
        if (m_pinnedOutput[i] > maxProb) {
            maxProb = m_pinnedOutput[i];
            maxClass = i;
        }
    fmt::print("Predicted class: {}, probability: {:.6f}\n", maxClass, maxProb);
    fmt::print("Inference latency: {} us\n", duration.count());
}

double TensorRTModule::benchmark(int iterations) {
    if (m_pinnedInput == nullptr) {
        m_logger->log(Severity::kERROR,
                      "m_pinnedInput is null. Call infer() first or preload input.");
        return 0.0;
    }

    // warm-up
    for (int i = 0; i < 10; ++i)
        doInference();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
        doInference();
    auto end = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double fps = iterations / (totalMs / 1000.0);

    fmt::print("\n=== Benchmark ({} iterations) ===\n", iterations);
    fmt::print("Total time: {:.2f} ms\n", totalMs);
    fmt::print("Avg latency: {:.3f} ms\n", totalMs / iterations);
    fmt::print("FPS: {:.1f}\n", fps);

    return fps;
}
