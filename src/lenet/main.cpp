#include <fmt/core.h>
#include <fstream>
#include "NvInfer.h"
#include "logging.h"
#include <NvOnnxParser.h>
#include <string_view>
#include <chrono>

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            fmt::print(stderr,"Cuda failure: {} at {}\n", ret,__LINE__);             \
            abort();                                           \
        }                                                      \
    } while (0)


// Minist dataset input dimensions
static constexpr int INPUT_H = 32;
static constexpr int INPUT_W = 32;
static constexpr int OUTPUT_SIZE = 10;

static constexpr std::string_view INPUT_NAME = "input";
static constexpr std::string_view OUTPUT_NAME = "probabilities";

static constexpr std::string_view LENET_ONNX_PATH = "/home/jetson/Programs/tensorrt/tensorrt-practice/src/lenet/models/lenet5.onnx";

using namespace nvinfer1;
static Logger glogger(Severity::kINFO);






void parserModel(const std::string& onnxFilePath, INetworkDefinition* network) {
    // 创建 Parser
    auto parser = nvonnxparser::createParser(*network, glogger);
    if (!parser->parseFromFile(onnxFilePath.c_str(), static_cast<int>(Severity::kINFO)))
    {
        const auto msg = fmt::format("Failed to parse ONNX file: {}", onnxFilePath);
        glogger.log(Severity::kERROR, msg.c_str());
        return;
    }
    // 设置输入输出名称

    network->getInput(0)->setName(INPUT_NAME.data());
    network->getOutput(0)->setName(OUTPUT_NAME.data());
}



void doInference(
        std::unique_ptr<nvinfer1::IExecutionContext>& context, 
        float* input, 
        float* output, 
        int batchSize = 1) {
        
    const auto& engine = context->getEngine();
    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbIOTensors() == 2);

    // 根据输入输出名称获取输入输出索引，供后续绑定（enqueueV3要求）
    const char* inputName = INPUT_NAME.data();
    const char* outputName = OUTPUT_NAME.data();

    // 1. 分配输入输出缓冲区
    void* deviceInput{nullptr};
    void* deviceOutput{nullptr};

    CHECK(cudaMalloc(&deviceInput, batchSize * 1 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float)));
    // 2. 创建 CUDA 流,用于同步CUDA操作
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // 3. 将输入数据复制到 GPU
    CHECK(cudaMemcpyAsync(deviceInput, input, batchSize * 1 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    // 4. 执行推理

    context->setTensorAddress(inputName, deviceInput);
    context->setTensorAddress(outputName, deviceOutput);
    context->enqueueV3(stream);
    // 5. 将输出数据复制回 CPU
    CHECK(cudaMemcpyAsync(output, deviceOutput, batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    // 同步流，等待推理完成
    CHECK(cudaStreamSynchronize(stream));
    // 6. 释放资源
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));

}

auto createEngine(std::unique_ptr<nvinfer1::IBuilder> builder, std::unique_ptr<nvinfer1::INetworkDefinition> network, bool useDLA = true) -> std::unique_ptr<nvinfer1::ICudaEngine> {
    // 创建 BuilderConfig
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>{builder->createBuilderConfig()};

    // 配置 DLA（如果需要）
    if (useDLA)
    {
        if (builder->getNbDLACores() > 0)
        {   
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
            // DLA 要求FP16精度或者INT8精度
            config->setFlag(BuilderFlag::kFP16);
            config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
            config->setDLACore(0); // 使用第一个 DLA 核心
            glogger.log(Severity::kINFO, "Using DLA for inference.");
        }
        else
        {
            glogger.log(Severity::kWARNING, "DLA not available. Falling back to GPU.");
        }
    }

    // 配置工作空间大小
    size_t free{0}, total{0};
    CHECK(cudaMemGetInfo(&free, &total));
     glogger.log(Severity::kINFO, fmt::format("GPU memory - free: {} MB, total: {} MB", free / (1024 * 1024), total / (1024 * 1024)).c_str());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, free);
    // 构建引擎
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>{builder->buildEngineWithConfig(*network, *config)};
    return engine;
}



void checkEngine(nvinfer1::ICudaEngine* engine) {

    if (engine == nullptr) {
        glogger.log(Severity::kERROR, "engine is null.");
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

void serializeEngine(std::unique_ptr<nvinfer1::ICudaEngine> engine,std::string_view engineFilePath) {
    // 将引擎序列化到磁盘，这里需要IHostMemory对象来保存序列化后的引擎数据,其中保存二进制数据
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>{engine->serialize()};
    assert(serializedEngine);
    std::ofstream engineFile(std::string(engineFilePath), std::ios::binary);

    if (!engineFile)
    {
        glogger.log(Severity::kERROR, "Failed to open engine file for writing.");
        return;
    }
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();

    glogger.log(Severity::kINFO, fmt::format("Engine serialized to {}", engineFilePath).c_str());
}

void buildEngine(const std::string& onnxFilePath, bool useDLA = false) {
    // 创建 Builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>{nvinfer1::createInferBuilder(glogger)};
    // 创建 Network（显式 batch）

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>{builder->createNetworkV2(explicitBatch)};
    
    parserModel(onnxFilePath, network.get());
    
    auto engine = createEngine(std::move(builder), std::move(network), useDLA);
    checkEngine(engine.get());
    serializeEngine(std::move(engine),onnxFilePath.substr(0, onnxFilePath.find_last_of('.')) + ".engine");
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

    // 创建 runtime
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(glogger)};
    assert(runtime != nullptr);
    // 反序列化引擎
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>{runtime->deserializeCudaEngine(engineData.data(), engineSize)};
    assert(engine != nullptr);

    // 创建执行上下文
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>{engine->createExecutionContext()};
    assert(context != nullptr);

    // 准备输入数据，这里以全1输入为例子
    std::vector<float> input(INPUT_H * INPUT_W, 1.0f);
    std::vector<float> output(OUTPUT_SIZE, 0.0f);

    // 推理1000次，计算总时间和平均时间
    const int iterations = 1000;
    fmt::print("Iterations: {}\n",iterations);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
        doInference(context, input.data(), output.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    fmt::print("Total time: {} microseconds\n", duration.count());
    fmt::print("Average time: {} microseconds\n", duration.count() / iterations);

    // 打印结果：十个类别的概率,并且输出最大的为结果
    for (const auto& prob : output) {
        fmt::print("{:.6f} ", prob);
    }
    fmt::print("\n");

    // 分类结果
    int maxClass = 0;
    float maxProb = output[0];
    for (int i = 1; i < OUTPUT_SIZE; ++i)
    {
        if (output[i] > maxProb)
        {
            maxProb = output[i];
            maxClass = i;
        }
    }
    fmt::print("Predicted class: {}, probability: {:.6f}\n", maxClass, maxProb);

}

int main(int argc, char** argv) {
    fmt::print("[lenet] module scaffold is ready.\n");

    if (argc > 1 && std::string(argv[1]) == "build")
    {
        buildEngine(LENET_ONNX_PATH.data(), false);
    }
    else
    {   
        auto engineFilePath = std::string(LENET_ONNX_PATH.data()).substr(0, std::string(LENET_ONNX_PATH.data()).find_last_of('.')) + ".engine";
        performInference(engineFilePath.data());
    }
    

    
    return 0;
}
