#pragma once
#include "CommonTypes_C_API.h"

#ifdef __cplusplus
extern "C" {
#endif

// ===================================================================
// 日志管理接口（对应Logger类的核心功能）
// ===================================================================

// 初始化日志系统 - 对应 Logger::getInstance().initialize(logConfig)
AOI_API int AOI_CALL AOI_Logger_Initialize(const AOI_LogConfig* config);

// 关闭日志系统 - 对应 Logger::getInstance().shutdown()
AOI_API void AOI_CALL AOI_Logger_Shutdown();


#ifdef __cplusplus
}
#endif