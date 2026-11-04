#pragma once
#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include <cstdint>
#include "CommonTypes.h"

// ===================================================================
// Logger特定类型和常量
// ===================================================================

// Logger日志级别（简化版，4级）
enum class LogLevel : int {
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4
};

// 日志配置结构体
struct LogConfig {
    char logPath[AOI_MAX_LOG_PATH_LEN] = {0};
    LogLevel level = LogLevel::INFO;
    int maxFileSize = 10 * 1024 * 1024;  // 10MB
    int maxFiles = 5;
    bool enableConsole = true;
    bool enableFile = true;

    // Inline methods
    inline bool isValid() const noexcept {
        bool logPathValid = !(strlen(logPath) == 0 && enableFile);
        bool fileSizeValid = maxFileSize > 0;
        bool fileCountValid = maxFiles > 0;
        return logPathValid && fileSizeValid && fileCountValid;
    }

    inline void reset() noexcept {
        logPath[0] = '\0';
        level = LogLevel::INFO;
        maxFileSize = 10 * 1024 * 1024;
        maxFiles = 5;
        enableConsole = true;
        enableFile = true;
    }
};

// High-Performance Logger - Simplified 4-level design
class Logger {
public:
    // Singleton pattern
    static Logger& getInstance();

    // Initialize and cleanup - RAII design
    int initialize(const LogConfig& config) noexcept;
    void shutdown() noexcept;

    // Simplified formatted logging with file location (unified interface)
    void debug(const char* format, const char* file = nullptr, int line = 0, ...) noexcept;
    void info(const char* format, const char* file = nullptr, int line = 0, ...) noexcept;
    void warn(const char* format, const char* file = nullptr, int line = 0, ...) noexcept;
    void error(const char* format, const char* file = nullptr, int line = 0, ...) noexcept;

    // Convenience methods
    bool isEnabled(LogLevel level) const noexcept;
    void setLevel(LogLevel level) noexcept;

    // Performance monitoring
    size_t getLogCount() const noexcept;
    void resetLogCount() noexcept;

    // Force flush buffered logs
    void flush() noexcept;

    // Thread management for DLL environments
    void setThreadName(const std::string& name) noexcept;

    // RAII - Disable copy and move
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

private:
    Logger();
    ~Logger();

    // High-performance implementation
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

// Unified module-based logging macros (formatted only)
#define LOG_DEBUG(module, fmt, ...) do { \
    if (Logger::getInstance().isEnabled(LogLevel::DEBUG)) \
        Logger::getInstance().debug("[%s] " fmt, __FILE__, __LINE__, module, ##__VA_ARGS__); \
} while(0)

#define LOG_INFO(module, fmt, ...) do { \
    if (Logger::getInstance().isEnabled(LogLevel::INFO)) \
        Logger::getInstance().info("[%s] " fmt, __FILE__, __LINE__, module, ##__VA_ARGS__); \
} while(0)

#define LOG_WARN(module, fmt, ...) do { \
    if (Logger::getInstance().isEnabled(LogLevel::WARN)) \
        Logger::getInstance().warn("[%s] " fmt, __FILE__, __LINE__, module, ##__VA_ARGS__); \
} while(0)

#define LOG_ERROR(module, fmt, ...) do { \
    if (Logger::getInstance().isEnabled(LogLevel::ERROR)) \
        Logger::getInstance().error("[%s] " fmt, __FILE__, __LINE__, module, ##__VA_ARGS__); \
} while(0)

// Convenience aliases without module parameter for direct usage
#define LOG_DEBUGX(fmt, ...) LOG_DEBUG("GENERIC", fmt, ##__VA_ARGS__)
#define LOG_INFOX(fmt, ...) LOG_INFO("GENERIC", fmt, ##__VA_ARGS__)
#define LOG_WARNX(fmt, ...) LOG_WARN("GENERIC", fmt, ##__VA_ARGS__)
#define LOG_ERRORX(fmt, ...) LOG_ERROR("GENERIC", fmt, ##__VA_ARGS__)

// Thread naming helper for DLL environments
#define SET_THREAD_NAME(name) Logger::getInstance().setThreadName(name)