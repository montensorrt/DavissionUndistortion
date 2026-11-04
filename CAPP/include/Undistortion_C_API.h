#pragma once
#include "CommonTypes_C_API.h"

#ifdef __cplusplus
extern "C"
{
#endif

    // ===================================================================
    // Undistortion specific data structures
    // ===================================================================

#pragma pack(push, 1)

    // Undistortion config structure
    struct AOI_UndistortionConfig
    {
        char matrixPath[512] = {0}; // Interpolation matrix folder path
        int matrixWidth = 0;        // Weight matrix width
        int matrixHeight = 0;       // Weight matrix height
        int outputChannels = 1;     // Output channels (1=grayscale, 3=color)
    };

#pragma pack(pop)

    // ===================================================================
    // Undistortion C API - function interfaces
    // ===================================================================

    /**
     * @brief Create Undistortion instance
     * @return Return Undistortion instance pointer, nullptr if failed
     */
    AOI_API void *AOI_CALL AOI_Undistortion_Create();

    /**
     * @brief Destroy Undistortion instance
     * @param undistorter Undistortion instance pointer
     */
    AOI_API void AOI_CALL AOI_Undistortion_Destroy(void *undistorter);

    /**
     * @brief Set undistortion config
     * @param undistorter Undistortion instance pointer
     * @param config Config struct pointer
     * @return Error code, 0 means success
     */
    AOI_API int AOI_CALL AOI_Undistortion_SetConfig(
        void *undistorter,
        const AOI_UndistortionConfig *config);

    /**
     * @brief Check if fully configured
     * @param undistorter Undistortion instance pointer
     * @return 1 means configured, 0 means not configured
     */
    AOI_API int AOI_CALL AOI_Undistortion_IsConfigured(void *undistorter);

    /**
     * @brief Perform image undistortion
     * @param undistorter Undistortion instance pointer
     * @param inputImage Input image data pointer
     * @param width Input image width
     * @param height Input image height
     * @param outputImage Output image data pointer
     * @param result Result struct pointer (AOI_BaseResult*)
     * @return Error code, 0 means success
     */
    AOI_API int AOI_CALL AOI_Undistortion_Infer(
        void *undistorter,
        const uint8_t *inputImage,
        int width,
        int height,
        uint8_t *outputImage,
        AOI_BaseResult *result);

#ifdef __cplusplus
}
#endif