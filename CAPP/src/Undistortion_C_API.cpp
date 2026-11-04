#include "Undistortion_C_API.h"
#include "Undistortion.h"
#include <new>
#include <cstring>

// ===================================================================
// Undistortion C API Implementation
// ===================================================================

/**
 * @brief Create Undistortion instance
 */
AOI_API void *AOI_CALL AOI_Undistortion_Create()
{
    try
    {
        return new ImageUndistorter();
    }
    catch (...)
    {
        return nullptr;
    }
}

/**
 * @brief Destroy Undistortion instance
 */
AOI_API void AOI_CALL AOI_Undistortion_Destroy(void *undistorter)
{
    if (undistorter)
    {
        delete static_cast<ImageUndistorter *>(undistorter);
    }
}

/**
 * @brief Set undistortion config
 */
AOI_API int AOI_CALL AOI_Undistortion_SetConfig(
    void *undistorter,
    const AOI_UndistortionConfig *config)
{
    if (!undistorter || !config)
    {
        return static_cast<int>(ErrorCode::INVALID_INPUT);
    }
    try
    {
        ImageUndistorter *ud = static_cast<ImageUndistorter *>(undistorter);
        // Convert C struct to C++ struct
        UndistortionConfig cppConfig;
        std::memset(&cppConfig, 0, sizeof(cppConfig));
        std::strncpy(cppConfig.matrixPath, config->matrixPath, sizeof(cppConfig.matrixPath) - 1);
        cppConfig.matrixWidth = config->matrixWidth;
        cppConfig.matrixHeight = config->matrixHeight;
        cppConfig.outputChannels = config->outputChannels;
        ErrorCode result = ud->setConfig(cppConfig);
        return static_cast<int>(result);
    }
    catch (...)
    {
        return static_cast<int>(ErrorCode::ERROR);
    }
}

/**
 * @brief Check if fully configured
 */
AOI_API int AOI_CALL AOI_Undistortion_IsConfigured(void *undistorter)
{
    if (!undistorter)
    {
        return 0;
    }
    try
    {
        ImageUndistorter *ud = static_cast<ImageUndistorter *>(undistorter);
        return ud->isConfigured() ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

/**
 * @brief Perform image undistortion
 */
AOI_API int AOI_CALL AOI_Undistortion_Infer(
    void *undistorter,
    const uint8_t *inputImage,
    int width,
    int height,
    uint8_t *outputImage,
    AOI_BaseResult *result)
{
    if (!undistorter || !inputImage || !outputImage || !result || width <= 0 || height <= 0)
    {
        return static_cast<int>(ErrorCode::INVALID_INPUT);
    }
    // Clear result struct
    std::memset(result, 0, sizeof(AOI_BaseResult));
    try
    {
        ImageUndistorter *ud = static_cast<ImageUndistorter *>(undistorter);
        BaseResult cppResult;
        ErrorCode errorCode = ud->infer(inputImage, width, height, outputImage, cppResult);
        // Convert result
        result->errorCode = static_cast<int>(errorCode);
        result->processingTime = cppResult.processingTime;
        std::snprintf(result->errorMessage, sizeof(result->errorMessage), "%s", cppResult.errorMessage);
        return static_cast<int>(errorCode);
    }
    catch (...)
    {
        result->errorCode = static_cast<int>(ErrorCode::ERROR);
        std::snprintf(result->errorMessage, sizeof(result->errorMessage), "%s", "Internal processing error");
        return static_cast<int>(ErrorCode::ERROR);
    }
}