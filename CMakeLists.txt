cmake_minimum_required(VERSION 3.22)
project(trt_practice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Shared, configurable dependency paths. Override via -D... when needed.
set(TRT_INCLUDE_DIR "/usr/include/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH "TensorRT include directory")
set(TRT_LIBRARY_DIR "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH "TensorRT library directory")

set(CUDA_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA root directory")
set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include" CACHE PATH "CUDA include directory")
set(CUDA_LIBRARY_DIR "${CUDA_ROOT_DIR}/lib64" CACHE PATH "CUDA library directory")

add_library(trt_practice_deps INTERFACE)
target_include_directories(trt_practice_deps SYSTEM INTERFACE
	"${TRT_INCLUDE_DIR}"
	"${CUDA_INCLUDE_DIR}"
)
target_link_directories(trt_practice_deps INTERFACE
	"${TRT_LIBRARY_DIR}"
	"${CUDA_LIBRARY_DIR}"
)

if(NOT EXISTS "${TRT_INCLUDE_DIR}")
	message(WARNING "TRT_INCLUDE_DIR does not exist: ${TRT_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${TRT_LIBRARY_DIR}")
	message(WARNING "TRT_LIBRARY_DIR does not exist: ${TRT_LIBRARY_DIR}")
endif()


find_package(fmt CONFIG REQUIRED)

add_subdirectory(src)