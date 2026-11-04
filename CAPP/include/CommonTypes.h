#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include "CommonTypes_C_API.h"

// ===================================================================
// 版本信息
// ===================================================================
#define AOI_UNDISTORTION_VERSION_MAJOR 1
#define AOI_UNDISTORTION_VERSION_MINOR 0
#define AOI_UNDISTORTION_VERSION_PATCH 0


// ===================================================================
// 简化后的ErrorCode - 只保留4项
// ===================================================================
enum class ErrorCode : int {
    INFER_SUCCESS = 0,  // 推理成功
    ALL_SUCCESS = 1,    // 全部成功
    INVALID_INPUT = 2,  // 输入无效
    ERROR = 3           // 通用错误
};

// ===================================================================
// 基础结果基类
// ===================================================================
struct BaseResult {
    ErrorCode errorCode = ErrorCode::INFER_SUCCESS;
    char errorMessage[AOI_MAX_ERROR_MSG_LEN] = {0};
    float processingTime = 0.0f;

    // Inline methods
    inline bool isSuccess() const noexcept {
        return errorCode == ErrorCode::INFER_SUCCESS || errorCode == ErrorCode::ALL_SUCCESS;
    }

    inline void reset() noexcept {
        errorCode = ErrorCode::INFER_SUCCESS;
        errorMessage[0] = '\0';
        processingTime = 0.0f;
    }
};
