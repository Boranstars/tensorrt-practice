#include <NvInferImpl.h>
#include <cassert>
#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <string>
#include <string_view>
#include <fmt/core.h>
#include <fstream>
#include <vector>
#include <chrono>
#include "logging.h"

using namespace nvinfer1;
static Logger glogger(Severity::kINFO);

constexpr int INPUT_SIZE = 1;
constexpr int OUTPUT_SIZE = 1;
std::unique_ptr<nvinfer1::ICudaEngine> createEngine(std::unique_ptr<nvinfer1::IBuilder> builder, std::unique_ptr<nvinfer1::INetworkDefinition> network, bool useDLA = true) {
    // 创建 BuilderConfig
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>{builder->createBuilderConfig()};

    // 设置设备
    if (useDLA) {
        glogger.log(Severity::kINFO, "Using DLA for inference.");
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
        config->setFlag(BuilderFlag::kFP16);
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(0);
    } else {
        glogger.log(Severity::kINFO, "Using GPU for inference.");
    }
    
    // 此处设置工作空间大小为当前GPU剩余内存大小
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    // 以MB为单位打印剩余内存和总内存
    glogger.log(Severity::kINFO, fmt::format("GPU memory - free: {} MB, total: {} MB", free / (1024 * 1024), total / (1024 * 1024)).c_str());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, free);
    // 构建引擎
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>{builder->buildEngineWithConfig(*network, *config)};

    assert(engine != nullptr);
    // 在构建完engine之后,builder和network就不再需要了,可以释放掉(通过智能指针自动释放，这里获取了所有权)
    return engine;
}

void checkEngine(nvinfer1::ICudaEngine* engine) {

    if (engine == nullptr) {
        glogger.log(Severity::kERROR, "Failed to build engine.");
        return;
    }
    int nbBindings = engine->getNbIOTensors();
    glogger.log(Severity::kINFO, fmt::format("Number of bindings: {}", nbBindings).c_str());
    for (int i = 0; i < nbBindings; i++)
    {
        auto tenorName = engine->getIOTensorName(i);
        glogger.log(Severity::kINFO, fmt::format("Binding {}: {}", i, tenorName).c_str());


    }
    int nbLayers = engine->getNbLayers();
    glogger.log(Severity::kINFO, fmt::format("Number of layers: {}", nbLayers).c_str());


}

void doInference(std::unique_ptr<nvinfer1::IExecutionContext>& context, float* input, float* output,[[maybe_unused]] int batchSize = 1) {
    // 推理步骤：

    const auto& engine = context->getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbIOTensors() number of buffers.
    assert(engine.getNbIOTensors() == 2);

    // 根据输入输出名称获取输入输出索引，供后续绑定（enqueueV3要求）
    // 这里要和序列化时设置的相同
    const char* inputName = "input";
    const char* outputName = "output";

    // 1. 分配输入输出缓冲区
    void* deviceInput{nullptr};
    void* deviceOutput{nullptr};

    cudaMalloc(&deviceInput, batchSize * INPUT_SIZE * sizeof(float));
    cudaMalloc(&deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float));
    // 2. 创建 CUDA 流,用于同步CUDA操作
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 3. 将输入数据复制到 GPU
    cudaMemcpyAsync(deviceInput, input, batchSize * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream);
    // 4. 执行推理

    // TRT10+ 推荐使用 enqueueV3 接口, 需要传入绑定tensor名称和CUDA流
    context->setTensorAddress(inputName, deviceInput);
    context->setTensorAddress(outputName, deviceOutput);
    context->enqueueV3(stream);
    

    // 5. 将输出数据复制回 CPU
    cudaMemcpyAsync(output, deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream);
    // 同步流，等待推理完成
    cudaStreamSynchronize(stream);

    // 6. 释放资源
    cudaStreamDestroy(stream);
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}

void serializeEngine(std::unique_ptr<nvinfer1::ICudaEngine> engine) {
    // 将引擎序列化到磁盘，这里需要IHostMemory对象来保存序列化后的引擎数据,其中保存二进制数据
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>{engine->serialize()};
    assert(serializedEngine);
    std::ofstream engineFile("mlp.engine", std::ios::binary);

    if (!engineFile)
    {
        glogger.log(Severity::kERROR, "Failed to open engine file for writing.");
        return;
    }
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    glogger.log(Severity::kINFO, "Engine serialized to mlp.engine");
}

void performInference() {
    // 推理步骤：
    // 1. 从磁盘加载引擎
    // 2. 创建runtime,用于反序列化引擎
    // 3. 反序列化引擎
    // 4. 创建执行上下文，用于执行推理
    // 5. 执行推理
    // 6. 处理输出结果


    // 从磁盘加载引擎
    std::ifstream engineFile("mlp.engine", std::ios::binary);
    if (!engineFile)
    {
        glogger.log(Severity::kERROR, "Failed to open engine file for reading.");
        return;
    }
    engineFile.seekg(0, std::ios::end);
    size_t engineSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(engineSize);
    engineFile.read(engineData.data(), engineSize);
    engineFile.close();

    
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(glogger)};

    assert(runtime != nullptr);
    // 反序列化引擎
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>{runtime->deserializeCudaEngine(engineData.data(), engineSize)};

    assert(engine != nullptr);
    // 这个，不需要了
    engineData.clear();
    engineData.shrink_to_fit();

    // 创建执行上下文
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>{engine->createExecutionContext()};
    assert(context != nullptr);

    // 执行推理

    glogger.log(Severity::kINFO, "Performing inference...");
    // 输入输出数据：
    float input[1] = {12.0f}; // 输入数据
    float output[1] = {0.0f}; // 输出数据

    // 计时：
    auto start = std::chrono::high_resolution_clock::now();

    doInference(context, input, output, 1);

    // 推理结束，计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inferenceTime = end - start;
    glogger.log(Severity::kINFO, fmt::format("Inference completed in {} seconds.", inferenceTime.count()).c_str());

    // 打印结果：
    fmt::print("Inference input: {}\n", input[0]);
    fmt::print("Inference result: {}\n", output[0]);
    

}

int main() {
    glogger.log(Severity::kINFO, "mlp module scaffold is ready.");

    // 模型构建步骤
    // 1. 生成ONNX模型
    // 2. 使用TensorRT的ONNX解析器将ONNX模型转换为TensorRT引擎
    // 3. 使用TensorRT引擎进行推理

    // 整体流程：
    // 创建 Builder
    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(glogger)};


    
    // 创建 Network（显式 batch）

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
    // 创建 Parser

    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, glogger)};
    // parseFromFile 解析 onnx
    constexpr std::string_view onnx_path = MLP_ONNX_PATH;
    if (!parser->parseFromFile(onnx_path.data(), static_cast<int>(Severity::kINFO)))
    {
        const auto msg = fmt::format("Failed to parse ONNX file: {}", onnx_path);
        glogger.log(Severity::kERROR, msg.c_str());
        return -1;
    }
    // 设置输入输出名称

    network->getInput(0)->setName("input");
    network->getOutput(0)->setName("output");



    // 后续可继续 build engine

    // network->markOutput(*network->getOutput(0)); // 标记输出节点，是否需要？
    
    glogger.log(Severity::kINFO, "------------------- Engine Build ------------------");

    auto engine = createEngine(std::move(builder), std::move(network));
    checkEngine(engine.get());
    serializeEngine(std::move(engine));

    glogger.log(Severity::kINFO, "------------------- Engine Serialized ------------------");
    
    performInference();

    return 0;
}
