


add_executable(yolov5_demo
    main.cpp
    tensorrt_module.cpp
    utils.cpp
)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)

target_include_directories(yolov5_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov5_demo PRIVATE
    trt_practice_deps
    nvinfer
    nvonnxparser
    cudart
    fmt::fmt
    ${OpenCV_LIBS}
)
