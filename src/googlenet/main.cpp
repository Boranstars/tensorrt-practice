#include "logging.h"
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <chrono>
#include <fmt/core.h>
#include <fstream>
#include <memory>
#include <string_view>
#include <vector>

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      fmt::print(stderr, "Cuda failure: {} at {}:{}\n", ret, __FILE__,         \
                 __LINE__);                                                    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

static constexpr std::string_view GOOGLENET_ONNX_PATH =
    "/home/jetson/Programs/tensorrt/tensorrt-practice/src/googlenet/models/"
    "googlenet.onnx";

using namespace nvinfer1;

struct TRTBuildConfig {
  bool useDLA{true};
  bool useFP16{true};
  int batchSize{1};
  int input_h{224};
  int input_w{224};
  int output_size{1000};
};

class TensorRTModule {
public:
  TensorRTModule(const std::string_view onnxFilePath,
                 std::unique_ptr<nvinfer1::ILogger> logger,
                 const TRTBuildConfig &config = TRTBuildConfig())
      : m_logger(std::move(logger)), m_onnxFilePath(onnxFilePath),
        m_config(config) {

    this->batchSize = config.batchSize;
    this->input_h = config.input_h;
    this->input_w = config.input_w;
    this->output_size = config.output_size;
  }

  ~TensorRTModule() {}

private:
  std::unique_ptr<nvinfer1::ILogger> m_logger;
  std::string m_onnxFilePath;
  TRTBuildConfig m_config;


  void *deviceInput{nullptr};
  void *deviceOutput{nullptr};
  std::string inputTensorName;
  std::string outputTensorName;
  int batchSize{1};
  int channels{3};
  int input_h{224};
  int input_w{224};
  int output_size{1000};

private:
  void parserModel(const std::string &onnxFilePath,
                   INetworkDefinition *network) {
    // 创建 Parser
    auto parser = nvonnxparser::createParser(*network, *m_logger);
    if (!parser->parseFromFile(onnxFilePath.c_str(),
                               static_cast<int>(Severity::kINFO))) {
      const auto msg =
          fmt::format("Failed to parse ONNX file: {}", onnxFilePath);
      m_logger->log(Severity::kERROR, msg.c_str());
      return;
    }
  }

  auto createEngine(std::unique_ptr<nvinfer1::IBuilder> builder,
                    std::unique_ptr<nvinfer1::INetworkDefinition> network)
      -> std::unique_ptr<nvinfer1::ICudaEngine> {
    // 创建 BuilderConfig

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>{
        builder->createBuilderConfig()};

    // 配置 DLA（如果需要）
    if (m_config.useDLA) {
      if (builder->getNbDLACores() > 0) {
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
        // DLA 要求FP16精度或者INT8精度
        config->setFlag(BuilderFlag::kFP16);
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(0); // 使用第一个 DLA 核心
        m_logger->log(Severity::kINFO, "Using DLA for inference.");
      } else {
        m_logger->log(Severity::kWARNING,
                      "DLA not available. Fallingback to GPU.");
      }
    }

    // 配置工作空间大小
    size_t free{0}, total{0};
    CHECK(cudaMemGetInfo(&free, &total));
    m_logger->log(Severity::kINFO,
                  fmt::format("GPU memory - free: {} MB, total: {} MB",
                              free / (1024 * 1024), total / (1024 * 1024))
                      .c_str());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, free);

    // 对于googlenet，最后一层由logit层改为softmax层,添加softmax层
    auto output = network->getOutput(0);
    network->unmarkOutput(*output);
    m_logger->log(
        Severity::kINFO,
        fmt::format("Original output tensor name: {}", output->getName())
            .c_str());
    auto softmax = network->addSoftMax(*output);
    softmax->setAxes(1 << 1); // 对输出轴做softmax
    softmax->getOutput(0)->setName("softmax_output");
    network->markOutput(*softmax->getOutput(0));

    m_logger->log(Severity::kINFO, fmt::format("New output tensor name: {}",
                                               softmax->getOutput(0)->getName())
                                       .c_str());
    // 打印新的输入输出信息
    int nbDims = network->getOutput(0)->getDimensions().nbDims;
    m_logger->log(
        Severity::kINFO,
        fmt::format("Output tensor dimensions: nbDims: {}, dims: ", nbDims)
            .c_str());

    // abort();
    // 构建引擎
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>{
        builder->buildEngineWithConfig(*network, *config)};
    return engine;
  }

  void serializeEngine(std::unique_ptr<nvinfer1::ICudaEngine> engine,
                       std::string_view engineFilePath) {
    // 将引擎序列化到磁盘，这里需要IHostMemory对象来保存序列化后的引擎数据,其中保存二进制数据
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

  void doInference(std::unique_ptr<nvinfer1::IExecutionContext> &context,
                   float *input, float *output, int batchSize = 1) {

    const auto &engine = context->getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbIOTensors() == 2);

    // 根据输入输出名称获取输入输出索引，供后续绑定（enqueueV3要求）
    const char *inputName = inputTensorName.data();
    const char *outputName = outputTensorName.data();

    // 1. 分配输入输出缓冲区
    void *deviceInput{nullptr};
    void *deviceOutput{nullptr};

    CHECK(cudaMalloc(&deviceInput,
                     batchSize * channels * input_h * input_w * sizeof(float)));
    CHECK(cudaMalloc(&deviceOutput, batchSize * output_size * sizeof(float)));
    // 2. 创建 CUDA 流,用于同步CUDA操作
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // 3. 将输入数据复制到 GPU
    CHECK(cudaMemcpyAsync(deviceInput, input,
                          batchSize * channels * input_h * input_w *
                              sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    // 4. 执行推理

    context->setTensorAddress(inputName, deviceInput);
    context->setTensorAddress(outputName, deviceOutput);
    bool success = context->enqueueV3(stream);
    if (!success) {
      m_logger->log(Severity::kERROR, "Failed to execute inference.");
      return;
    }
    // 5. 将输出数据复制回 CPU
    CHECK(cudaMemcpyAsync(output, deviceOutput,
                          batchSize * output_size * sizeof(float),
                          cudaMemcpyDeviceToHost, stream));
    // 同步流，等待推理完成
    CHECK(cudaStreamSynchronize(stream));
    // 6. 释放资源
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));
  }

  void performInference(std::string_view engineFilePath) {
    // 推理步骤：
    // 1. 从磁盘加载引擎
    // 2. 创建runtime,用于反序列化引擎
    // 3. 反序列化引擎
    // 4. 创建执行上下文，用于执行推理
    // 5. 执行推理
    // 6. 处理输出结果

    // 从磁盘加载引擎
    std::ifstream engineFile(engineFilePath.data(), std::ios::binary);
    if (!engineFile) {
      m_logger->log(Severity::kERROR,
                    "Failed to open engine file for reading.");
      return;
    }
    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    // 创建 runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>{
        nvinfer1::createInferRuntime(*m_logger)};
    assert(runtime != nullptr);
    // 反序列化引擎
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>{
        runtime->deserializeCudaEngine(engineData.data(), engineSize)};
    assert(engine != nullptr);

    // 获取输入输出信息
    this->inputTensorName = engine->getIOTensorName(0);
    this->outputTensorName = engine->getIOTensorName(1);
    fmt::print("Input tensor name: {}, Output tensor name: {}\n",
               inputTensorName, outputTensorName);

    // 创建执行上下文
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>{
        engine->createExecutionContext()};
    assert(context != nullptr);

    // 准备输入数据，这里以全1输入为例子
    std::vector<float> input(batchSize * channels * input_h * input_w, 1.0f);
    std::vector<float> output(batchSize * output_size, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    doInference(context, input.data(), output.data());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 打印结果：十个类别的概率,并且输出最大的为结果
    for (const auto &prob : output) {
      fmt::print("{:.6f} ", prob);
    }
    fmt::print("\n");

    // 分类结果
    int maxClass = 0;
    float maxProb = output[0];
    for (int i = 1; i < output_size; ++i) {
      if (output[i] > maxProb) {
        maxProb = output[i];
        maxClass = i;
      }
    }
    fmt::print("Predicted class: {}, probability: {:.6f}\n", maxClass, maxProb);
  }

public:
  void initialize(const TRTBuildConfig &buildConfig = TRTBuildConfig()) {

    // 检查是否存在序列化的引擎文件，如果存在则跳过构建步骤
    std::string engineFilePath =
        m_onnxFilePath.substr(0, m_onnxFilePath.find_last_of('.')) + ".engine";
    std::ifstream engineFile(engineFilePath);
    if (engineFile.good()) {
      m_logger->log(Severity::kINFO,
                    fmt::format("Serialized engine file {} already exists. "
                                "Skipping engine build.",
                                engineFilePath)
                        .c_str());
      return;
    }

    // 1. 创建 Builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>{
        nvinfer1::createInferBuilder(*m_logger)};
    assert(builder != nullptr);

    // 2. 创建 NetworkDefinition
    const auto explicitBatch =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>{
        builder->createNetworkV2(0U)};
    assert(network != nullptr);

    // 3. 解析 ONNX 模型
    parserModel(m_onnxFilePath.data(), network.get());

    // 4. 构建引擎
    auto engine = createEngine(std::move(builder), std::move(network));

    // 5. 序列化引擎到磁盘
    serializeEngine(std::move(engine),
                    m_onnxFilePath.substr(0, m_onnxFilePath.find_last_of('.')) +
                        ".engine");
  }

  void infer() {
    performInference(
        m_onnxFilePath.substr(0, m_onnxFilePath.find_last_of('.')) + ".engine");
  }
};

int main() {
  // Placeholder entry for module scaffolding only.
  fmt::print("googlenet module scaffold is ready.\n");

  Logger logger(Severity::kINFO);
  TensorRTModule module(GOOGLENET_ONNX_PATH.data(),
                        std::make_unique<Logger>(logger));
  module.initialize();
  module.infer();
  return 0;
}
