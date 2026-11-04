#include "Logger.h"
#include <iostream>
#include <cstdarg>
#include <memory>
#include <mutex>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <thread>
#include <vector>
#include <condition_variable>
#include <unordered_map>

// Enhanced log entry with thread ID and file location information
struct LogEntry {
    LogLevel level;
    std::chrono::high_resolution_clock::time_point timestamp;
    std::thread::id threadId;           // Thread ID for multi-thread debugging
    char fileName[128];                 // Source file name (extracted)
    int lineNumber;                     // Source line number
    char message[1024];                // Fixed size to avoid dynamic allocation

    LogEntry() : level(LogLevel::INFO),
                 timestamp(std::chrono::high_resolution_clock::now()),
                 threadId(std::this_thread::get_id()),
                 lineNumber(0) {
        fileName[0] = '\0';
        message[0] = '\0';
    }

    LogEntry(LogLevel lvl, const char* msg,
             const char* file = nullptr, int line = 0)
        : level(lvl),
          timestamp(std::chrono::high_resolution_clock::now()),
          threadId(std::this_thread::get_id()),
          lineNumber(line) {

        // Copy message
        strncpy(message, msg, sizeof(message) - 1);
        message[sizeof(message) - 1] = '\0';

        // Extract just the filename, not full path
        if (file) {
            const char* lastSlash = strrchr(file, '/');
            const char* lastBackslash = strrchr(file, '\\');
            const char* filename = (lastSlash > lastBackslash) ? (lastSlash + 1) :
                                  (lastBackslash ? (lastBackslash + 1) : file);
            strncpy(fileName, filename, sizeof(fileName) - 1);
            fileName[sizeof(fileName) - 1] = '\0';
        } else {
            fileName[0] = '\0';
        }
    }
};

// Enhanced Logger implementation with thread ID and file location support
struct Logger::Impl {
    LogConfig config_;
    std::mutex mutex_;
    std::ofstream fileStream_;
    std::atomic<bool> initialized_{false};

    // Performance optimizations
    std::atomic<LogLevel> currentLevel_{LogLevel::INFO};
    std::atomic<size_t> logCount_{0};

    // Time caching - update every 10ms
    std::atomic<uint64_t> cachedTimeValue_{0};
    std::string cachedTimeString_;
    std::thread timeUpdateThread_;
    std::atomic<bool> timeThreadRunning_{false};

    // Thread name management for better debugging
    std::unordered_map<std::thread::id, std::string> threadNames_;
    mutable std::mutex threadNamesMutex_;

    // Batch writing - buffer 100 logs
    std::vector<LogEntry> logBuffer_;
    static constexpr size_t BUFFER_SIZE = 100;
    std::thread writerThread_;
    std::condition_variable writerCV_;
    std::mutex bufferMutex_;
    std::atomic<bool> writerThreadRunning_{false};
    bool forceFlush_{false};

    Impl() : logBuffer_(BUFFER_SIZE) {
        logBuffer_.clear();
    }

    ~Impl() {
        shutdown();
    }

    int initialize(const LogConfig& config) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            if (initialized_) {
                shutdown();
            }

            config_ = config;
            currentLevel_.store(config.level);

            if (!config_.isValid()) {
                std::cerr << "Logger: Invalid configuration" << std::endl;
                return -1;
            }

            // Open file if needed
            if (config_.enableFile && strlen(config_.logPath) > 0) {
                fileStream_.open(config_.logPath, std::ios::app);
                if (!fileStream_.is_open()) {
                    std::cerr << "Logger: Failed to open log file: " << config_.logPath << std::endl;
                    return -2;
                }
            }

            // Start time update thread
            startTimeUpdateThread();

            // Start batch writer thread if file logging is enabled
            if (config_.enableFile) {
                startWriterThread();
            }

            initialized_ = true;

            // Log initialization message
            if (config_.enableConsole) {
                std::cout << "Logger: Enhanced logger initialized with thread ID and file location support" << std::endl;
            }
            if (config_.enableFile && fileStream_.is_open()) {
                logDirect(LogLevel::INFO, "Logger: Enhanced logger initialized with thread ID and file location support", __FILE__, __LINE__);
            }

            return 0;
        }
        catch (const std::exception& e) {
            std::cerr << "Logger: Initialization exception: " << e.what() << std::endl;
            return -3;
        }
        catch (...) {
            std::cerr << "Logger: Unknown initialization exception" << std::endl;
            return -4;
        }
    }

    void shutdown() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        try {
            if (initialized_) {
                // Stop threads
                stopTimeUpdateThread();
                stopWriterThread();

                // Flush remaining logs
                flushBuffer();

                // Log shutdown message
                if (config_.enableConsole) {
                    std::cout << "Logger: Shutting down" << std::endl;
                }
                if (config_.enableFile && fileStream_.is_open()) {
                    logDirect(LogLevel::INFO, "Logger: Shutting down", __FILE__, __LINE__);
                }

                if (fileStream_.is_open()) {
                    fileStream_.close();
                }

                initialized_ = false;
            }
        }
        catch (...) {
            // Ignore exceptions during shutdown
        }
    }

    // Thread name management
    void setThreadName(const std::string& name) noexcept {
        std::lock_guard<std::mutex> lock(threadNamesMutex_);
        threadNames_[std::this_thread::get_id()] = name;
    }

    std::string getThreadName(std::thread::id threadId) const {
        std::lock_guard<std::mutex> lock(threadNamesMutex_);
        auto it = threadNames_.find(threadId);
        if (it != threadNames_.end()) {
            return it->second;
        }
        return "";
    }

    void startTimeUpdateThread() {
        timeThreadRunning_ = true;
        timeUpdateThread_ = std::thread([this]() {
            while (timeThreadRunning_) {
                updateTimeCache();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }

    void stopTimeUpdateThread() {
        if (timeThreadRunning_) {
            timeThreadRunning_ = false;
            if (timeUpdateThread_.joinable()) {
                timeUpdateThread_.join();
            }
        }
    }

    void startWriterThread() {
        writerThreadRunning_ = true;
        writerThread_ = std::thread([this]() {
            std::unique_lock<std::mutex> lock(bufferMutex_);
            while (writerThreadRunning_) {
                writerCV_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return !logBuffer_.empty() || forceFlush_ || !writerThreadRunning_;
                });

                if (!logBuffer_.empty()) {
                    writeBatchToFile();
                }
                forceFlush_ = false;
            }
        });
    }

    void stopWriterThread() {
        if (writerThreadRunning_) {
            writerThreadRunning_ = false;
            forceFlush_ = true;
            writerCV_.notify_all();
            if (writerThread_.joinable()) {
                writerThread_.join();
            }
        }
    }

    void updateTimeCache() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();

        cachedTimeString_ = ss.str();
        cachedTimeValue_.store(static_cast<uint64_t>(now.time_since_epoch().count()));
    }

    const std::string& getCachedTimeString() const {
        return cachedTimeString_;
    }

    // Enhanced log method with file location
    void log(LogLevel level, const char* message,
             const char* file = nullptr, int line = 0) noexcept {
        if (!initialized_ || !message || level < currentLevel_.load()) {
            return;
        }

        logCount_++;
        LogEntry entry(level, message, file, line);

        // Console output (immediate)
        if (config_.enableConsole) {
            logToConsole(entry);
        }

        // File output (buffered)
        if (config_.enableFile) {
            std::lock_guard<std::mutex> lock(bufferMutex_);
            logBuffer_.push_back(entry);

            if (logBuffer_.size() >= BUFFER_SIZE) {
                writerCV_.notify_one();
            }
        }
    }

    // Core formatted log method
    void log(LogLevel level, const char* format,
              const char* file, int line, va_list args) noexcept {
        if (!initialized_ || !format || level < currentLevel_.load()) {
            return;
        }

        try {
            char buffer[1024];
            int result = vsnprintf(buffer, sizeof(buffer), format, args);

            if (result > 0 && result < static_cast<int>(sizeof(buffer))) {
                log(level, buffer, file, line);
            }
            else {
                log(LogLevel::ERROR, "Log formatting error: message too long or format error", file, line);
            }
        }
        catch (...) {
            log(LogLevel::ERROR, "Log formatting exception", file, line);
        }
    }

    void logToConsole(const LogEntry& entry) const {
        const std::string& timeStr = getCachedTimeString();
        const char* levelStr = getLevelString(entry.level);

        // Get thread ID as string
        std::ostringstream threadIdStream;
        threadIdStream << std::hex << entry.threadId;
        std::string threadId = threadIdStream.str();

        // Get thread name if available
        std::string threadName = getThreadName(entry.threadId);
        std::string threadInfo = threadName.empty() ? threadId : threadName + "(" + threadId + ")";

        // Enhanced format: time [level] [thread] [file:line] message
        std::string logLine = timeStr + " [" + levelStr + "] [T:" + threadInfo + "]";
        if (entry.lineNumber > 0 && strlen(entry.fileName) > 0) {
            logLine += " [" + std::string(entry.fileName) + ":" + std::to_string(entry.lineNumber) + "]";
        }
        logLine += " " + std::string(entry.message);

        if (entry.level >= LogLevel::ERROR) {
            std::cerr << logLine << std::endl;
        }
        else {
            std::cout << logLine << std::endl;
        }
    }

    void logDirect(LogLevel level, const char* message,
                   const char* file = nullptr, int line = 0) {
        if (fileStream_.is_open()) {
            const std::string& timeStr = getCachedTimeString();
            const char* levelStr = getLevelString(level);

            // Get thread ID as string
            std::ostringstream threadIdStream;
            threadIdStream << std::hex << std::this_thread::get_id();
            std::string threadId = threadIdStream.str();

            // Extract file name
            char fileName[128] = {0};
            if (file) {
                const char* lastSlash = strrchr(file, '/');
                const char* lastBackslash = strrchr(file, '\\');
                const char* filename = (lastSlash > lastBackslash) ? (lastSlash + 1) :
                                      (lastBackslash ? (lastBackslash + 1) : file);
                strncpy(fileName, filename, sizeof(fileName) - 1);
            }

            fileStream_ << timeStr << " [" << levelStr << "] [T:" << threadId << "]";
            if (line > 0 && strlen(fileName) > 0) {
                fileStream_ << " [" << fileName << ":" << line << "]";
            }
            fileStream_ << " " << message << std::endl;
            fileStream_.flush();
        }
    }

    void writeBatchToFile() {
        if (!fileStream_.is_open() || logBuffer_.empty()) {
            return;
        }

        try {
            for (const auto& entry : logBuffer_) {
                const std::string& timeStr = getCachedTimeString();
                const char* levelStr = getLevelString(entry.level);

                // Get thread ID as string
                std::ostringstream threadIdStream;
                threadIdStream << std::hex << entry.threadId;
                std::string threadId = threadIdStream.str();

                // Get thread name if available
                std::string threadName = getThreadName(entry.threadId);
                std::string threadInfo = threadName.empty() ? threadId : threadName + "(" + threadId + ")";

                // Enhanced format: time [level] [thread] [file:line] message
                fileStream_ << timeStr << " [" << levelStr << "] [T:" + threadInfo + "]";
                if (entry.lineNumber > 0 && strlen(entry.fileName) > 0) {
                    fileStream_ << " [" << entry.fileName << ":" << entry.lineNumber << "]";
                }
                fileStream_ << " " << entry.message << std::endl;
            }
            fileStream_.flush();
            logBuffer_.clear();
        }
        catch (...) {
            // Ignore file writing exceptions
        }
    }

    void flushBuffer() {
        if (config_.enableFile) {
            std::lock_guard<std::mutex> lock(bufferMutex_);
            if (!logBuffer_.empty()) {
                writeBatchToFile();
            }
        }
    }

    const char* getLevelString(LogLevel level) const {
        switch (level) {
        case LogLevel::DEBUG:  return "DEBUG";
        case LogLevel::INFO:   return "INFO";
        case LogLevel::WARN:   return "WARN";
        case LogLevel::ERROR:  return "ERROR";
        default:                        return "UNKNOWN";
        }
    }

    bool isEnabled(LogLevel level) const noexcept {
        return initialized_ && level >= currentLevel_.load();
    }

    void setLevel(LogLevel level) noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        currentLevel_.store(level);
    }

    size_t getLogCount() const noexcept {
        return logCount_.load();
    }

    void resetLogCount() noexcept {
        logCount_ = 0;
    }
};

// Enhanced Logger implementation methods
Logger::Logger() : pImpl_(std::make_unique<Impl>()) {
}

Logger::~Logger() = default;

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

int Logger::initialize(const LogConfig& config) noexcept {
    return pImpl_ ? pImpl_->initialize(config) : -1;
}

void Logger::shutdown() noexcept {
    if (pImpl_) {
        pImpl_->shutdown();
    }
}

// Enhanced logging methods with file location
// Simplified formatted logging methods (unified interface)
void Logger::debug(const char* format, const char* file, int line, ...) noexcept {
    if (pImpl_) {
        va_list args;
        va_start(args, format);
        pImpl_->log(LogLevel::DEBUG, format, file, line, args);
        va_end(args);
    }
}

void Logger::info(const char* format, const char* file, int line, ...) noexcept {
    if (pImpl_) {
        va_list args;
        va_start(args, format);
        pImpl_->log(LogLevel::INFO, format, file, line, args);
        va_end(args);
    }
}

void Logger::warn(const char* format, const char* file, int line, ...) noexcept {
    if (pImpl_) {
        va_list args;
        va_start(args, format);
        pImpl_->log(LogLevel::WARN, format, file, line, args);
        va_end(args);
    }
}

void Logger::error(const char* format, const char* file, int line, ...) noexcept {
    if (pImpl_) {
        va_list args;
        va_start(args, format);
        pImpl_->log(LogLevel::ERROR, format, file, line, args);
        va_end(args);
    }
}

bool Logger::isEnabled(LogLevel level) const noexcept {
    return pImpl_ ? pImpl_->isEnabled(level) : false;
}

void Logger::setLevel(LogLevel level) noexcept {
    if (pImpl_) {
        pImpl_->setLevel(level);
    }
}

size_t Logger::getLogCount() const noexcept {
    return pImpl_ ? pImpl_->getLogCount() : 0;
}

void Logger::resetLogCount() noexcept {
    if (pImpl_) {
        pImpl_->resetLogCount();
    }
}

void Logger::flush() noexcept {
    if (pImpl_) {
        pImpl_->flushBuffer();
    }
}

// Thread name management
void Logger::setThreadName(const std::string& name) noexcept {
    if (pImpl_) {
        pImpl_->setThreadName(name);
    }
}