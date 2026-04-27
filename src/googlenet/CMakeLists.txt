add_executable(googlenet_demo
    main.cpp
)

target_link_libraries(googlenet_demo PRIVATE
    trt_practice_deps
    nvinfer
    nvonnxparser
    cudart
    fmt::fmt
)
