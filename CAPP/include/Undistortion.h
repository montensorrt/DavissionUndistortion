#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <cstring>
#include "CommonTypes.h"

// ===================================================================
// Undistortion特定配置
// ===================================================================

// Undistortion配置结构体
struct UndistortionConfig {
    char matrixPath[512] = {0};  // 插值矩阵文件夹路径（UTF-8编码）
    int matrixWidth = 0;         // 权重矩阵宽度
    int matrixHeight = 0;        // 权重矩阵高度
    int outputChannels = 0;      // 输出通道数（0=自动，1=灰度，3=彩色）

    // Inline methods
    inline bool isValid() const noexcept {
        bool pathValid = strlen(matrixPath) > 0;
        bool widthValid = matrixWidth > 0;
        bool heightValid = matrixHeight > 0;
        bool channelsValid = outputChannels == 0 || outputChannels == 1 || outputChannels == 3;
        return pathValid && widthValid && heightValid && channelsValid;
    }

    inline void reset() noexcept {
        matrixPath[0] = '\0';
        matrixWidth = 0;
        matrixHeight = 0;
        outputChannels = 0;
    }
};

// ===================================================================
// PIMPL Design - Forward Declaration
// ===================================================================
class ImageUndistorterImpl;

// ===================================================================
// Main Interface Class - C++ Internal Use (Object-Oriented)
// ===================================================================
class ImageUndistorter
{
public:
    // RAII - Construction and destruction
    ImageUndistorter();
    ~ImageUndistorter();

    // 禁用拷贝，允许移动
    ImageUndistorter(const ImageUndistorter &) = delete;
    ImageUndistorter &operator=(const ImageUndistorter &) = delete;
    ImageUndistorter(ImageUndistorter &&) noexcept;
    ImageUndistorter &operator=(ImageUndistorter &&) noexcept;

    // 初始化（加载权重矩阵等）
    ErrorCode setConfig(const UndistortionConfig &config);
    // Check if configured
    bool isConfigured() const noexcept;
    // 主推理接口（图像去畸变）
    // inputImage/outputImage: pointer to externally allocated buffer, size=width*height*channels
    // stride: bytes per row, usually width*channels
    // result: use BaseResult directly
    ErrorCode infer(const uint8_t *inputImage, int width, int height, uint8_t *outputImage,
                    BaseResult &result) noexcept;

    // 性能基准测试接口
    // 运行多次推理并计算平均性能
    ErrorCode benchmark(const uint8_t *inputImage, int width, int height, uint8_t *outputImage,
                       int iterations, BaseResult &result) noexcept;

private:
    std::unique_ptr<ImageUndistorterImpl> pImpl_;
};
