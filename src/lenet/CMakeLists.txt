add_executable(lenet_demo
    main.cpp
)

target_link_libraries(lenet_demo PRIVATE
    trt_practice_deps
    nvinfer
    cudart
    nvonnxparser
    fmt::fmt
)
