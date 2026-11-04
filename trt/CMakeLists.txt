cmake_minimum_required(VERSION 3.20)
project(undistortion_v1 LANGUAGES CXX)

# ===================================================================
# 去畸变 V1 —— TensorRT 推理库 + C ABI 验收测试
# 参考 INFER/deploy/trt 的构建方式（Ninja + MSVC + 显式 TensorRT 路径）
# ===================================================================

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED TENSORRT_ROOT)
    if(DEFINED ENV{TENSORRT_ROOT})
        set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
    else()
        set(TENSORRT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v10.14.1.48")
    endif()
endif()
if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found under: ${TENSORRT_ROOT}/include")
endif()

# OpenCV 只用于测试程序的 BMP 读写与逐像素比对；DLL 本体不依赖 OpenCV
if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "E:/3rdparty/opencv/build")
endif()
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/utf-8 /MP /wd4828 /wd4819)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /MD")
endif()

# -------------------------------------------------------------------
# undist_api.dll —— 唯一交付物（C ABI）
# -------------------------------------------------------------------
add_library(undist_api SHARED
    src/undist_tables.cpp
    src/undist_engine.cpp
    src/undist_api.cpp
)

target_include_directories(undist_api
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${TENSORRT_ROOT}/include ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(undist_api PRIVATE "${TENSORRT_ROOT}/lib")
target_link_libraries(undist_api PRIVATE nvinfer_10 CUDA::cudart)
target_compile_definitions(undist_api PRIVATE UNDIST_API_EXPORTS)

# 只导出 undist_api.h 里显式 __declspec(dllexport) 的 C 符号；
# 不要用 WINDOWS_EXPORT_ALL_SYMBOLS，否则 Engine/undist_ctx 的 C++ 符号会一起泄漏。
set_target_properties(undist_api PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# -------------------------------------------------------------------
# test_undist_api.exe —— 验收 + 性能 + 内存模式对照
# -------------------------------------------------------------------
add_executable(test_undist_api tests/test_undist_api.cpp)
target_include_directories(test_undist_api PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_undist_api PRIVATE undist_api ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(test_undist_api PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

message(STATUS "===========================================")
message(STATUS "undistortionV1 native build")
message(STATUS "  Build Type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  TensorRT   : ${TENSORRT_ROOT}")
message(STATUS "  CUDA       : ${CUDAToolkit_VERSION}")
message(STATUS "  OpenCV     : ${OpenCV_VERSION}")
message(STATUS "===========================================")
