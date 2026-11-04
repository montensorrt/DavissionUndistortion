#include "Logger_C_API.h"
#include "Logger.h"
#include <cstring>

extern "C"
{

    // ===================================================================
    // 日志管理核心接口实现
    // ===================================================================

    AOI_API int AOI_CALL AOI_Logger_Initialize(const AOI_LogConfig *config)
    {
        if (!config)
        {
            return 1; // 错误：无效输入
        }

        try
        {
            // 转换AOI_LogConfig -> LogConfig（直接字段赋值）
            LogConfig cppConfig;
            strncpy_s(cppConfig.logPath, sizeof(cppConfig.logPath), config->logPath, _TRUNCATE);
            cppConfig.level = static_cast<LogLevel>(config->level);
            cppConfig.maxFileSize = config->maxFileSize;
            cppConfig.maxFiles = config->maxFiles;
            cppConfig.enableConsole = (config->enableConsole != 0);
            cppConfig.enableFile = (config->enableFile != 0);

            // 调用C++接口
            Logger &logger = Logger::getInstance();
            int result = logger.initialize(cppConfig);

            return result;
        }
        catch (...)
        {
            return 2; // 错误：初始化失败
        }
    }

    AOI_API void AOI_CALL AOI_Logger_Shutdown()
    {
        try
        {
            Logger &logger = Logger::getInstance();
            logger.shutdown();
        }
        catch (...)
        {
            // 静默处理异常
        }
    }

} // extern "C"