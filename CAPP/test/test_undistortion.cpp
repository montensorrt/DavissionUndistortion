#include "Undistortion.h"
#include "Logger.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <ctime>

// ===================================================================
// 配置常量
// ===================================================================
constexpr char RAW_INPUT_PATH[] = "E:/algrithm/project/2DAOI/feipai_input/product_2/1022/143323/raw_grouped";
constexpr char MATRIX_PATH[] = "E:/algrithm/project/2DAOI/feipai_input/undistort/2DAOI";
constexpr char OUTPUT_DIR[] = "E:/algrithm/project/2DAOI/feipai_input/product_2/1022/143323/undistorted";

constexpr int MATRIX_WIDTH = 5120;
constexpr int MATRIX_HEIGHT = 4096;
constexpr int OUTPUT_CHANNELS = 1;

// 图像扩展名
const std::vector<std::string> IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif"};

// ===================================================================
// 工具函数
// ===================================================================

// 生成带时间戳的日志文件名
std::string GenerateTimestampLogFile() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm_now;
    localtime_s(&tm_now, &time_t_now);

    std::ostringstream oss;
    oss << "undistortion_test_" << std::setfill('0') << std::setw(4) << tm_now.tm_year + 1900
        << std::setw(2) << tm_now.tm_mon + 1 << std::setw(2) << tm_now.tm_mday << "_"
        << std::setw(2) << tm_now.tm_hour << std::setw(2) << tm_now.tm_min
        << std::setw(2) << tm_now.tm_sec << ".log";
    return oss.str();
}

// 检查文件是否为图像文件
bool IsImageFile(const std::string& extension) {
    return std::find(IMAGE_EXTENSIONS.begin(), IMAGE_EXTENSIONS.end(), extension) != IMAGE_EXTENSIONS.end();
}

// 递归遍历目录，获取所有图像文件
void RecursiveFindImages(const std::string& rootPath, std::vector<std::string>& imagePaths) {
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(rootPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (IsImageFile(ext)) {
                    imagePaths.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error traversing directory: " << e.what() << std::endl;
    }
}

// 获取相对路径（从输入根目录开始）- 修复迭代器不兼容问题
std::string GetRelativePath(const std::string& rootPath, const std::string& fullPath) {
    std::filesystem::path root(rootPath);
    std::filesystem::path full(fullPath);

    // 使用相同的path对象进行迭代器操作，避免不兼容问题
    auto rootIt = root.begin();
    auto fullIt = full.begin();
    auto rootEnd = root.end();
    auto fullEnd = full.end();

    // 找到公共部分
    while (rootIt != rootEnd && fullIt != fullEnd && *rootIt == *fullIt) {
        ++rootIt;
        ++fullIt;
    }

    std::string relative;
    for (; fullIt != fullEnd; ++fullIt) {
        if (!relative.empty()) relative += "/";
        relative += fullIt->string();
    }
    return relative;
}

// 确保输出目录存在
void EnsureOutputDirectory(const std::string& path) {
    try {
        std::filesystem::create_directories(path);
    } catch (const std::exception& e) {
        std::cerr << "Error creating directory " << path << ": " << e.what() << std::endl;
    }
}

// 初始化日志系统
void InitializeLogger() {
    LogConfig config;
    config.level = LogLevel::DEBUG;  // 修改为DEBUG级别以查看详细日志
    config.maxFileSize = 10 * 1024 * 1024; // 10MB
    config.maxFiles = 5;
    config.enableConsole = true;
    config.enableFile = true;

    std::string logFile = GenerateTimestampLogFile();
    strncpy(config.logPath, logFile.c_str(), sizeof(config.logPath) - 1);

    Logger& logger = Logger::getInstance();
    int result = logger.initialize(config);
    if (result != 0) {
        std::cerr << "Failed to initialize logger: " << result << std::endl;
    }

    std::cout << "=== Undistortion Test Started ===" << std::endl;
    std::cout << "Log file: " << logFile << std::endl;
    std::cout << "Input path: " << RAW_INPUT_PATH << std::endl;
    std::cout << "Matrix path: " << MATRIX_PATH << std::endl;
    std::cout << "Output path: " << OUTPUT_DIR << std::endl;
    LOG_INFO("TEST", "Logger initialized successfully");
}

// 关闭日志系统
void ShutdownLogger() {
    std::cout << "\n=== Test Completed ===" << std::endl;
    LOG_INFO("TEST", "Test suite completed");
    Logger::getInstance().shutdown();
}

// 图像顺时针旋转90度
cv::Mat RotateClockwise90(const cv::Mat& image) {
    cv::Mat rotated;
    cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
    return rotated;
}

// 图像逆时针旋转90度
cv::Mat RotateCounterClockwise90(const cv::Mat& image) {
    cv::Mat rotated;
    cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
    return rotated;
}

// 统计耗时，去除最大最小值并计算平均
float CalculateAverageWithoutMinMax(const std::vector<float>& times) {
    if (times.size() < 3) {
        // 如果少于3个值，计算简单平均
        if (times.empty()) return 0.0f;
        return std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    }

    std::vector<float> sortedTimes = times;
    std::sort(sortedTimes.begin(), sortedTimes.end());

    // 去除最小值和最大值
    float sum = 0.0f;
    for (size_t i = 1; i < sortedTimes.size() - 1; ++i) {
        sum += sortedTimes[i];
    }

    return sum / (sortedTimes.size() - 2);
}

// 保存结果到指定路径
void SaveResult(const cv::Mat& image, const std::string& relativePath, const std::string& outputDir) {
    std::filesystem::path outputPath = std::filesystem::path(outputDir) / relativePath;
    EnsureOutputDirectory(outputPath.parent_path().string());

    // 使用u8string()处理Unicode路径
    std::string pathStr = outputPath.u8string();

    if (!cv::imwrite(pathStr, image)) {
        std::cerr << "Failed to save image: " << pathStr << std::endl;
        std::string errorMsg = "Failed to save image: " + pathStr;
        LOG_ERROR("TEST", "%s", errorMsg.c_str());
    } else {
        std::cout << "Saved: " << pathStr << std::endl;
        std::string successMsg = "Saved result: " + pathStr;
        LOG_INFO("TEST", "%s", successMsg.c_str());
    }
}

// ===================================================================
// 主测试流程
// ===================================================================
int main() {
    try {
        // 1. 初始化日志系统
        InitializeLogger();

        // 2. 递归加载所有图像文件
        LOG_INFO("TEST", "Starting to find images recursively...");
        std::vector<std::string> imagePaths;
        RecursiveFindImages(RAW_INPUT_PATH, imagePaths);

        if (imagePaths.empty()) {
            LOG_ERROR("TEST", "No images found in input path");
            std::cerr << "No images found in: " << RAW_INPUT_PATH << std::endl;
            ShutdownLogger();
            return -1;
        }

        std::cout << "Found " << imagePaths.size() << " images" << std::endl;
        LOG_INFO("TEST", "Found %zu images", imagePaths.size());

        // 3. 初始化去畸变句柄
        LOG_INFO("TEST", "Initializing undistortion handler...");
        ImageUndistorter undistorter;
        UndistortionConfig config;
        strncpy(config.matrixPath, MATRIX_PATH, sizeof(config.matrixPath) - 1);
        config.matrixWidth = MATRIX_WIDTH;
        config.matrixHeight = MATRIX_HEIGHT;
        config.outputChannels = OUTPUT_CHANNELS;

        ErrorCode initResult = undistorter.setConfig(config);
        if (initResult != ErrorCode::INFER_SUCCESS) {
            LOG_ERROR("TEST", "Failed to initialize undistortion handler");
            std::cerr << "Failed to initialize undistortion handler" << std::endl;
            ShutdownLogger();
            return -1;
        }

        std::cout << "Undistortion handler initialized successfully" << std::endl;
        LOG_INFO("TEST", "Undistortion handler initialized successfully");

        // 确保输出目录存在
        EnsureOutputDirectory(OUTPUT_DIR);

        // 4-8. 处理所有图像，统计各步骤耗时
        std::vector<float> rotateClockwiseTimes;
        std::vector<float> undistortTimes;
        std::vector<float> rotateCounterClockwiseTimes;

        BaseResult result;
        int processedCount = 0;
        int failedCount = 0;

        for (const auto& imagePath : imagePaths) {
            try {
                // 加载图像
                cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << imagePath << std::endl;
                    LOG_ERROR("TEST", "Failed to load image: %s", imagePath.c_str());
                    failedCount++;
                    continue;
                }

                // 5. 顺时针旋转90度并计时
                auto startRotateClockwise = std::chrono::high_resolution_clock::now();
                cv::Mat rotatedClockwise = RotateClockwise90(image);
                auto endRotateClockwise = std::chrono::high_resolution_clock::now();
                float rotateClockwiseTime = std::chrono::duration<float, std::milli>(
                    endRotateClockwise - startRotateClockwise).count();
                rotateClockwiseTimes.push_back(rotateClockwiseTime);

                // 6. 去畸变处理并计时
                std::vector<uint8_t> inputData(rotatedClockwise.data, rotatedClockwise.data + rotatedClockwise.total());
                std::vector<uint8_t> outputData(rotatedClockwise.cols * rotatedClockwise.rows * OUTPUT_CHANNELS);

                auto startUndistort = std::chrono::high_resolution_clock::now();
                ErrorCode undistortResult = undistorter.infer(inputData.data(), rotatedClockwise.cols, rotatedClockwise.rows,
                                                             outputData.data(), result);
                auto endUndistort = std::chrono::high_resolution_clock::now();

                if (undistortResult != ErrorCode::INFER_SUCCESS) {
                    std::cerr << "Undistortion failed for: " << imagePath << std::endl;
                    LOG_ERROR("TEST", "Undistortion failed for: %s", imagePath.c_str());
                    failedCount++;
                    continue;
                }

                float undistortTime = std::chrono::duration<float, std::milli>(
                    endUndistort - startUndistort).count();
                undistortTimes.push_back(undistortTime);

                // 7. 逆时针旋转90度并计时
                cv::Mat undistortedMat(rotatedClockwise.rows, rotatedClockwise.cols, CV_8UC1, outputData.data());
                auto startRotateCounterClockwise = std::chrono::high_resolution_clock::now();
                cv::Mat finalImage = RotateCounterClockwise90(undistortedMat);
                auto endRotateCounterClockwise = std::chrono::high_resolution_clock::now();
                float rotateCounterClockwiseTime = std::chrono::duration<float, std::milli>(
                    endRotateCounterClockwise - startRotateCounterClockwise).count();
                rotateCounterClockwiseTimes.push_back(rotateCounterClockwiseTime);

                // 10. 保存结果（保持相对路径结构）
                std::string relativePath = GetRelativePath(RAW_INPUT_PATH, imagePath);
                SaveResult(finalImage, relativePath, OUTPUT_DIR);

                processedCount++;

                // 打印进度
                if (processedCount % 10 == 0 || processedCount == imagePaths.size()) {
                    std::cout << "Processed: " << processedCount << "/" << imagePaths.size() << std::endl;
                    LOG_INFO("TEST", "Progress: %zu/%zu", processedCount, imagePaths.size());
                }

            } catch (const std::exception& e) {
                std::cerr << "Error processing " << imagePath << ": " << e.what() << std::endl;
                LOG_ERROR("TEST", "Error processing %s: %s", imagePath.c_str(), e.what());
                failedCount++;
            }
        }

        // 9. 计算各步骤平均耗时（去除最大最小值）
        float avgRotateClockwise = CalculateAverageWithoutMinMax(rotateClockwiseTimes);
        float avgUndistort = CalculateAverageWithoutMinMax(undistortTimes);
        float avgRotateCounterClockwise = CalculateAverageWithoutMinMax(rotateCounterClockwiseTimes);

        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << "Total images processed: " << processedCount << "/" << imagePaths.size() << std::endl;
        std::cout << "Failed: " << failedCount << std::endl;
        std::cout << "\nAverage processing times (excluding min/max):" << std::endl;
        std::cout << "  Clockwise rotation (90°): " << std::fixed << std::setprecision(3)
                  << avgRotateClockwise << " ms" << std::endl;
        std::cout << "  Undistortion: " << std::fixed << std::setprecision(3)
                  << avgUndistort << " ms" << std::endl;
        std::cout << "  Counter-clockwise rotation (90°): " << std::fixed << std::setprecision(3)
                  << avgRotateCounterClockwise << " ms" << std::endl;
        std::cout << "  Total: " << std::fixed << std::setprecision(3)
                  << (avgRotateClockwise + avgUndistort + avgRotateCounterClockwise) << " ms" << std::endl;

        LOG_INFO("TEST", "Performance Summary - Total: %zu/%zu", processedCount, imagePaths.size());
        LOG_INFO("TEST", "Avg Clockwise rotation: %.2f ms", avgRotateClockwise);
        LOG_INFO("TEST", "Avg Undistortion: %.2f ms", avgUndistort);
        LOG_INFO("TEST", "Avg Counter-clockwise rotation: %.2f ms", avgRotateCounterClockwise);
        LOG_INFO("TEST", "Total avg time: %.2f ms", avgRotateClockwise + avgUndistort + avgRotateCounterClockwise);

        ShutdownLogger();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        std::string errorMsg = "Fatal error: " + std::string(e.what());
        LOG_ERROR("TEST", "%s", errorMsg.c_str());
        ShutdownLogger();
        return -1;
    }
}