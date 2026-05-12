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
    if (m_graphExec)
        cudaGraphExecDestroy(m_graphExec);
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

void TensorRTModule::checkEngine(nvinfer1::ICudaEngine *engine) 
{

    if (engine == nullptr) {
        m_logger->log(Severity::kERROR, "engine is null.");
        return;
    }
    int nbBindings = engine->getNbIOTensors();
    m_logger->log(Severity::kINFO, fmt::format("Number of bindings: {}", nbBindings).c_str());
    for (int i = 0; i < nbBindings; i++)
    {
        auto tenorName = engine->getIOTensorName(i);
        m_logger->log(Severity::kINFO, fmt::format("Binding {}: {}", i, tenorName).c_str());


    }
    int nbLayers = engine->getNbLayers();
    m_logger->log(Severity::kINFO, fmt::format("Number of layers: {}", nbLayers).c_str());


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

bool TensorRTModule::doInference() {
    if (m_graphExec) {
        const cudaError_t launchStatus = cudaGraphLaunch(m_graphExec, m_stream);
        if (launchStatus == cudaSuccess)
            return true;

        m_logger->log(Severity::kWARNING,
                      fmt::format("CUDA Graph launch failed: {}. Falling back to enqueueV3.",
                                  cudaGetErrorString(launchStatus))
                          .c_str());
    }

    if (!m_context->enqueueV3(m_stream)) {
        m_logger->log(Severity::kERROR, "Failed to execute inference.");
        return false;
    }
    return true;
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
    checkEngine(m_engine.get());
    m_context.reset(m_engine->createExecutionContext());
    assert(m_context != nullptr);

    inputTensorName = m_engine->getIOTensorName(0);
    outputTensorName = m_engine->getIOTensorName(1);

    const size_t inputSize = m_params.batchSize * m_params.channels *
                              m_params.input_h * m_params.input_w * sizeof(float);
    const size_t outputSize = m_params.batchSize * m_params.output_size * sizeof(float);

    void* rawInput  = nullptr;
    void* rawOutput = nullptr;
    CHECK(cudaHostAlloc(&rawInput,  inputSize,  cudaHostAllocMapped | cudaHostAllocWriteCombined));
    CHECK(cudaHostAlloc(&rawOutput, outputSize, cudaHostAllocMapped));
    m_hostInput  = CudaHostPtr{rawInput,  cudaFreeHost};
    m_hostOutput = CudaHostPtr{rawOutput, cudaFreeHost};
    CHECK(cudaHostGetDevicePointer(&deviceInput,  m_hostInput.get(),  0));
    CHECK(cudaHostGetDevicePointer(&deviceOutput, m_hostOutput.get(), 0));
    CHECK(cudaStreamCreate(&m_stream));

    m_context->setTensorAddress(inputTensorName.c_str(), deviceInput);
    m_context->setTensorAddress(outputTensorName.c_str(), deviceOutput);

    if (!buildGraph()) {
        m_logger->log(Severity::kWARNING,
                      "CUDA Graph build failed. Falling back to normal enqueueV3.");
    }

    m_logger->log(Severity::kINFO,
                  "TensorRTModule initialized. Resources allocated.");
}

bool TensorRTModule::infer(const float* input, size_t numElements) {
    const size_t expected = static_cast<size_t>(m_params.batchSize) * m_params.channels *
                            m_params.input_h * m_params.input_w;
    if (input == nullptr) {
        m_logger->log(Severity::kERROR, "input is null.");
        return false;
    }
    if (numElements != expected) {
        m_logger->log(Severity::kERROR,
                      fmt::format("infer: expected {} elements, got {}.", expected, numElements).c_str());
        return false;
    }
    std::memcpy(m_hostInput.get(), input, expected * sizeof(float));
    if (!doInference())
        return false;

    const cudaError_t syncStatus = cudaStreamSynchronize(m_stream);
    if (syncStatus != cudaSuccess) {
        m_logger->log(Severity::kERROR,
                      fmt::format("Inference stream synchronization failed: {}",
                                  cudaGetErrorString(syncStatus))
                          .c_str());
        return false;
    }
    return true;
}

double TensorRTModule::benchmark(int iterations, const float* inputData) {
    if (inputData == nullptr) {
        m_logger->log(Severity::kERROR, "inputData is null.");
        return 0.0;
    }

    const size_t inputSize = static_cast<size_t>(m_params.batchSize) *
                              m_params.channels * m_params.input_h * m_params.input_w;
    std::memcpy(m_hostInput.get(), inputData, inputSize * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!doInference())
            return 0.0;

        const cudaError_t syncStatus = cudaStreamSynchronize(m_stream);
        if (syncStatus != cudaSuccess) {
            m_logger->log(Severity::kERROR,
                          fmt::format("Benchmark stream synchronization failed: {}",
                                      cudaGetErrorString(syncStatus))
                              .c_str());
            return 0.0;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double totalMs = std::chrono::duration<double, std::milli>(end - start).count();
    double fps = iterations / (totalMs / 1000.0);

    fmt::print("\n=== Benchmark ({} iterations) ===\n", iterations);
    fmt::print("Total time: {:.2f} ms\n", totalMs);
    fmt::print("Avg latency: {:.3f} ms\n", totalMs / iterations);
    fmt::print("FPS: {:.1f}\n", fps);

    return fps;
}

bool TensorRTModule::buildGraph() {
    m_logger->log(Severity::kINFO, "Starting CUDA Graph capture...");
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }

    cudaGraph_t graph{nullptr}; // 这是一个临时的“蓝图”

    const auto logCudaWarning = [this](const char* step, cudaError_t status) {
        m_logger->log(Severity::kWARNING,
                      fmt::format("CUDA Graph {} failed: {}", step,
                                  cudaGetErrorString(status))
                          .c_str());
    };

    cudaError_t status = cudaStreamSynchronize(m_stream);
    if (status != cudaSuccess) {
        logCudaWarning("pre-capture synchronization", status);
        return false;
    }

    // 1. 开机：开始录制
    // cudaStreamCaptureModeGlobal 表示录制期间，这个流上的所有异步操作都会被抓取
    status = cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeGlobal);
    if (status != cudaSuccess) {
        logCudaWarning("begin capture", status);
        return false;
    }

    // 2. 演戏：执行推理 API (只准有异步操作！)
    if (!m_context->enqueueV3(m_stream)) {
        m_logger->log(Severity::kERROR, "Failed to enqueue during graph capture.");
        status = cudaStreamEndCapture(m_stream, &graph);
        if (status == cudaSuccess && graph)
            cudaGraphDestroy(graph);
        return false;
    }

    // 3. 关机：结束录制，蓝图保存在 graph 中
    status = cudaStreamEndCapture(m_stream, &graph);
    if (status != cudaSuccess) {
        logCudaWarning("end capture", status);
        if (graph)
            cudaGraphDestroy(graph);
        return false;
    }

    // 4. 冲洗底片：把蓝图实例化为真正的“可执行文件” (m_graphExec)
    // 这一步比较耗时，所以我们放在初始化阶段做
    status = cudaGraphInstantiate(&m_graphExec, graph, 0);
    if (status != cudaSuccess) {
        logCudaWarning("instantiate", status);
        cudaGraphDestroy(graph);
        m_graphExec = nullptr;
        return false;
    }

    // 5. 销毁蓝图（因为已经有可执行文件了）
    status = cudaGraphDestroy(graph);
    if (status != cudaSuccess) {
        logCudaWarning("destroy temporary graph", status);
    }

    m_logger->log(Severity::kINFO, "CUDA Graph captured and instantiated successfully.");
    return true;
}