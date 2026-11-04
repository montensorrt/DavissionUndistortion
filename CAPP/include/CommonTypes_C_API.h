#pragma once
#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#ifdef AOI_EXPORTS
#define AOI_API __declspec(dllexport)
#else
#define AOI_API __declspec(dllimport)
#endif
#define AOI_CALL __cdecl
#else
#define AOI_API
#define AOI_CALL
#endif

// ===================================================================
// 版本信息
// ===================================================================

// C API版本
#define AOI_C_API_VERSION_MAJOR 1
#define AOI_C_API_VERSION_MINOR 0
#define AOI_C_API_VERSION_PATCH 0

// Undistortion版本
#define AOI_UNDISTORTION_VERSION_MAJOR 1
#define AOI_UNDISTORTION_VERSION_MINOR 0
#define AOI_UNDISTORTION_VERSION_PATCH 0

// ===================================================================
// 常量定义
// ===================================================================
#ifndef AOI_MAX_ERROR_MSG_LEN
constexpr size_t AOI_MAX_ERROR_MSG_LEN = 256;
#endif

#ifndef AOI_MAX_OBJECT_NAME_LEN
constexpr size_t AOI_MAX_OBJECT_NAME_LEN = 128;
#endif

#ifndef AOI_MAX_OBJECT_ID_LEN
constexpr size_t AOI_MAX_OBJECT_ID_LEN = 64;
#endif

#ifndef AOI_MAX_LOG_PATH_LEN
constexpr size_t AOI_MAX_LOG_PATH_LEN = 256;
#endif

// ===================================================================
// 确保跨语言内存对齐
// ===================================================================
#pragma pack(push, 1)

// ===================================================================
// 基础类型（C API通用类型）
// ===================================================================

// ===================================================================
// 基础结果类（对应BaseResult）
// ===================================================================

// 通用结果基类 - 对应 BaseResult（简化后的4项ErrorCode）
struct AOI_BaseResult
{
    int errorCode = 0; // 对应 ErrorCode::INFER_SUCCESS = 0
    char errorMessage[AOI_MAX_ERROR_MSG_LEN] = {0};
    float processingTime = 0.0f;
};

// ===================================================================
// 日志配置（对应Logger.h中的LogConfig）
// ===================================================================

// 日志配置结构 - 对应 LogConfig（简化版4级日志）
struct AOI_LogConfig
{
    char logPath[AOI_MAX_LOG_PATH_LEN] = {0};
    int level = 2;                      // 对应 LogLevel::INFO = 2
    int maxFileSize = 10 * 1024 * 1024; // 10MB
    int maxFiles = 5;
    int enableConsole = 1; // 对应 bool enableConsole = true
    int enableFile = 1;    // 对应 bool enableFile = true
};

#pragma pack(pop)

// ===================================================================
// 编译时类型大小验证（确保C#互操作兼容性）
// ===================================================================
#define AOI_C_VALIDATE_STRUCT_SIZE(StructName, ExpectedSize) \
    static_assert(sizeof(StructName) == ExpectedSize,        \
                  #StructName " size mismatch for C# interop")

// 基础类型大小验证 - temporarily commented out for debugging
// AOI_C_VALIDATE_STRUCT_SIZE(AOI_BaseResult, AOI_MAX_ERROR_MSG_LEN + 8); // errorMessage(256) + errorCode(4) + processingTime(4) = 264
// AOI_C_VALIDATE_STRUCT_SIZE(AOI_LogConfig, AOI_MAX_LOG_PATH_LEN + 17);   // logPath(256) + level(4) + maxFileSize(4) + maxFiles(4) + enableConsole(1) + enableFile(1) + padding(3) = 273

#ifdef __cplusplus
extern "C"
{
#endif

    // ===================================================================
    // 版本信息接口（唯一的工具函数）
    // ===================================================================

    AOI_API int AOI_CALL AOI_GetVersion(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif