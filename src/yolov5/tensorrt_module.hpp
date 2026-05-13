#pragma once

#include "logging.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#if defined(__cpp_lib_span)
#include <span>
#endif

using CudaHostPtr = std::unique_ptr<void, decltype(&cudaFreeHost)>;

struct TensorMetadata {
    int index{-1};
    std::string name;
    nvinfer1::TensorIOMode mode{nvinfer1::TensorIOMode::kNONE};
    nvinfer1::Dims dims{};
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};
    size_t elementCount{0};
    size_t byteSize{0};

    bool syncFromEngine(nvinfer1::ICudaEngine* engine,
                        nvinfer1::IExecutionContext* context,
                        int tensorIndex);
};

struct TRTParams {
    std::string onnxFilePath;
    bool useDLA{false};
    bool useFP16{true};
    int batchSize{1};
    int channels{3};
    int input_h{224};
    int input_w{224};
    int output_size{1000};
};

class TensorRTModule {
  public:
    TensorRTModule(std::unique_ptr<nvinfer1::ILogger> logger,
                   const TRTParams &config = TRTParams());
    ~TensorRTModule();

    void initialize(const TRTParams &buildConfig = TRTParams());

    // 核心接口：指针 + 元素数，内部校验长度
    bool infer(const float* input, size_t numElements);
    bool inferFromHostBuffer();

#if defined(__cpp_lib_span)
    // C++20：span 重载，自动携带长度
    bool infer(std::span<const float> input) {
        return infer(input.data(), input.size());
    }
#else
    // 回退：vector 重载
    bool infer(const std::vector<float>& input) {
        return infer(input.data(), input.size());
    }
#endif

    double benchmark(int iterations, const float* inputData);

    float* getHostInput()  const { return static_cast<float*>(m_hostInput.get()); }
    float* getHostOutput() const { return static_cast<float*>(m_hostOutput.get()); }
    // Current inference path supports exactly one input/output. Valid after initialize() succeeds.
    const TensorMetadata& getInputTensorMetadata() const {
        assert(!m_inputTensors.empty());
        return m_inputTensors.front();
    }
    const TensorMetadata& getOutputTensorMetadata() const {
        assert(!m_outputTensors.empty());
        return m_outputTensors.front();
    }
    const std::vector<TensorMetadata>& getInputTensorsMetadata() const { return m_inputTensors; }
    const std::vector<TensorMetadata>& getOutputTensorsMetadata() const { return m_outputTensors; }

  private:
    [[deprecated("Use the cli tool `trtexec` instead")]] void parserModel(const std::string &onnxFilePath,
                     nvinfer1::INetworkDefinition *network);
    [[deprecated("Use the cli tool `trtexec` instead")]] auto createEngine(std::unique_ptr<nvinfer1::IBuilder> builder,
                      std::unique_ptr<nvinfer1::INetworkDefinition> network)
        -> std::unique_ptr<nvinfer1::ICudaEngine>;
    [[deprecated("Use the cli tool `trtexec` instead")]] void serializeEngine(std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                         std::string_view engineFilePath);
    bool doInference();
    void checkEngine(nvinfer1::ICudaEngine *engine, nvinfer1::IExecutionContext *context);
    bool cacheEngineIO(nvinfer1::ICudaEngine *engine, nvinfer1::IExecutionContext *context);
    bool buildGraph();

  private:
    std::unique_ptr<nvinfer1::ILogger> m_logger;
    TRTParams m_params;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    CudaHostPtr m_hostInput{nullptr, cudaFreeHost};
    CudaHostPtr m_hostOutput{nullptr, cudaFreeHost};
    void *deviceInput{nullptr};
    void *deviceOutput{nullptr};
    cudaStream_t m_stream{nullptr};
    cudaGraphExec_t m_graphExec{nullptr};

    std::vector<TensorMetadata> m_inputTensors;
    std::vector<TensorMetadata> m_outputTensors;
};
