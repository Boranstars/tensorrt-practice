#include "tensorrt_module.hpp"

#include <NvOnnxParser.h>
#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
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

TensorRTModule::TensorRTModule(std::unique_ptr<nvinfer1::ILogger> logger,
                               const TRTParams &config)
    : m_logger(std::move(logger)), m_params(config) {}

TensorRTModule::~TensorRTModule() {
    if (deviceInput)
        cudaFree(deviceInput);
    if (deviceOutput)
        cudaFree(deviceOutput);
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

void TensorRTModule::doInference(float *input, float *output) {
    CHECK(cudaMemcpyAsync(deviceInput, input,
                          m_params.batchSize * m_params.channels * m_params.input_h *
                              m_params.input_w * sizeof(float),
                          cudaMemcpyHostToDevice, m_stream));
    if (!m_context->enqueueV3(m_stream))
        m_logger->log(Severity::kERROR, "Failed to execute inference.");
    CHECK(cudaMemcpyAsync(output, deviceOutput,
                          m_params.batchSize * m_params.output_size * sizeof(float),
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

    CHECK(cudaMalloc(&deviceInput,
                     m_params.batchSize * m_params.channels * m_params.input_h *
                         m_params.input_w * sizeof(float)));
    CHECK(cudaMalloc(&deviceOutput,
                     m_params.batchSize * m_params.output_size * sizeof(float)));
    CHECK(cudaStreamCreate(&m_stream));

    m_context->setTensorAddress(inputTensorName.c_str(), deviceInput);
    m_context->setTensorAddress(outputTensorName.c_str(), deviceOutput);

    m_logger->log(Severity::kINFO,
                  "TensorRTModule initialized. Resources allocated.");
}

void TensorRTModule::infer(const float* inputData) {
    if (inputData == nullptr) {
        m_logger->log(Severity::kERROR, "inputData is null.");
        return;
    }

    std::vector<float> input(m_params.batchSize * m_params.channels * m_params.input_h *
                                 m_params.input_w);
    std::copy(inputData, inputData + input.size(), input.begin());
    std::vector<float> output(m_params.batchSize * m_params.output_size, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    doInference(input.data(), output.data());
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start);

    for (const auto &prob : output)
        fmt::print("{:.6f} ", prob);
    fmt::print("\n");

    int maxClass = 0;
    float maxProb = output[0];
    for (int i = 1; i < m_params.output_size; ++i)
        if (output[i] > maxProb) {
            maxProb = output[i];
            maxClass = i;
        }
    fmt::print("Predicted class: {}, probability: {:.6f}\n", maxClass, maxProb);
    fmt::print("Inference latency: {} us\n", duration.count());
}

double TensorRTModule::benchmark(int iterations, const float* inputData) {
    if (inputData == nullptr) {
        m_logger->log(Severity::kERROR, "inputData is null.");
        return 0.0;
    }

    std::vector<float> input(m_params.batchSize * m_params.channels * m_params.input_h *
                                 m_params.input_w);
    std::copy(inputData, inputData + input.size(), input.begin());
    std::vector<float> output(m_params.batchSize * m_params.output_size, 0.0f);

    // warm-up
    for (int i = 0; i < 10; ++i)
        doInference(input.data(), output.data());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
        doInference(input.data(), output.data());
    auto end = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double fps = iterations / (totalMs / 1000.0);

    fmt::print("\n=== Benchmark ({} iterations) ===\n", iterations);
    fmt::print("Total time: {:.2f} ms\n", totalMs);
    fmt::print("Avg latency: {:.3f} ms\n", totalMs / iterations);
    fmt::print("FPS: {:.1f}\n", fps);

    return fps;
}
