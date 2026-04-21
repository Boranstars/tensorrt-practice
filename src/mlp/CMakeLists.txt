add_executable(mlp_demo
    main.cpp
    
)

target_compile_definitions(mlp_demo
    PRIVATE MLP_ONNX_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/mlp.onnx"
)

target_link_libraries(mlp_demo
    PRIVATE
        trt_practice_deps
        nvinfer
        nvonnxparser
    fmt::fmt
    cudart
)
