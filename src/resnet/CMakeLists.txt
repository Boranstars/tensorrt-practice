add_executable(resnet_demo
    main.cpp
    tensorrt_module.cpp
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

target_include_directories(resnet_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(resnet_demo PRIVATE
    trt_practice_deps
    nvinfer
    nvonnxparser
    cudart
    fmt::fmt
    ${OpenCV_LIBS}
)
