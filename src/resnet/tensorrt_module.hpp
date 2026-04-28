#pragma once

#include "logging.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct TRTParams {
    std::string onnxFilePath;
    bool useDLA{true};
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
    void infer(const std::string &imagePath);
    double benchmark(int iterations);

  private:
    void parserModel(const std::string &onnxFilePath,
                     nvinfer1::INetworkDefinition *network);
    auto createEngine(std::unique_ptr<nvinfer1::IBuilder> builder,
                      std::unique_ptr<nvinfer1::INetworkDefinition> network)
        -> std::unique_ptr<nvinfer1::ICudaEngine>;
    void serializeEngine(std::unique_ptr<nvinfer1::ICudaEngine> &engine,
                         std::string_view engineFilePath);
    void doInference();

  private:
    std::unique_ptr<nvinfer1::ILogger> m_logger;
    TRTParams m_params;

    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    void *deviceInput{nullptr};
    void *deviceOutput{nullptr};
    float *m_pinnedInput{nullptr};
    float *m_pinnedOutput{nullptr};
    size_t m_inputSize{0};
    size_t m_outputSize{0};
    cudaStream_t m_stream{nullptr};

    std::string inputTensorName;
    std::string outputTensorName;
};
