#include <NvInfer.h>
#include <fmt/core.h>
#include <memory>

#define CHECK(status)                                          \
    do {                                                       \
        auto ret = (status);                                   \
        if (ret != 0) {                                        \
            fmt::print(stderr,"Cuda failure: {} at {}\n", ret,__LINE__);             \
            abort();                                           \
        }                                                      \
    } while (0)


class TensorRTModule {
public:
    TensorRTModule(const std::string_view onnxFilePath, std::unique_ptr<nvinfer1::IBuilderConfig> config) {
        
    }

    ~TensorRTModule() {
        
    }


    
};
    




int main() {
    // Placeholder entry for module scaffolding only.
    fmt::print("googlenet module scaffold is ready.\n");
    return 0;
}
