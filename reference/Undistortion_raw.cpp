#include "Undistortion.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include "Logger.h"

// =============================
// UndistortionConfig Implementation
// =============================
bool UndistortionConfig::isValid() const noexcept
{
    LOG_DEBUGF("UndistortionConfig::isValid() - matrixPath=%s, matrixWidth=%d, matrixHeight=%d, outputChannels=%d", matrixPath, matrixWidth, matrixHeight, outputChannels); // Log config params
    bool valid = matrixPath[0] != 0 && (outputChannels == 0 || outputChannels == 1 || outputChannels == 3);
    LOG_DEBUGF("UndistortionConfig::isValid() - result: %d", valid); // Log result
    return valid;
}
void UndistortionConfig::reset() noexcept
{
    LOG_DEBUGF("UndistortionConfig::reset() - before: matrixPath=%s, matrixWidth=%d, matrixHeight=%d, outputChannels=%d", matrixPath, matrixWidth, matrixHeight, outputChannels); // Log before reset
    std::memset(this, 0, sizeof(*this));
    LOG_DEBUGF("UndistortionConfig::reset() - after: matrixPath=%s, matrixWidth=%d, matrixHeight=%d, outputChannels=%d", matrixPath, matrixWidth, matrixHeight, outputChannels); // Log after reset
}

// =============================
// ImageUndistorter PIMPL Implementation
// =============================
class ImageUndistorterImpl
{
public:
    ImageUndistorterImpl() = default;
    ~ImageUndistorterImpl() = default;

    /**
     * @brief Set undistortion config, load weight matrix and validate parameters
     */
    ErrorCode setConfig(const UndistortionConfig &config)
    {
        LOG_INFO("[Undistortion] call setConfig");
        LOG_DEBUGF("setConfig - matrixPath=%s, matrixWidth=%d, matrixHeight=%d, outputChannels=%d", config.matrixPath, config.matrixWidth, config.matrixHeight, config.outputChannels); // Log config params
        config_ = config;
        initialized_ = false;
        matrixWidth_ = config.matrixWidth;
        matrixHeight_ = config.matrixHeight;
        channels_ = (config.outputChannels == 0) ? 1 : config.outputChannels;
        inputStride_ = matrixWidth_ * channels_;
        outputStride_ = matrixWidth_ * channels_;
        // Config validation
        if (!config_.isValid())
        {
            LOG_ERROR("[Undistortion] Invalid config: matrixPath is empty or outputChannels is illegal");
            LOG_DEBUGF("Invalid config details: matrixPath=%s, outputChannels=%d", config_.matrixPath, config_.outputChannels); // Log invalid config
            return ErrorCode::INVALID_CONFIGURATION;
        }
        // Weight file validation
        std::string errorMsg;
        ErrorCode fileCheckCode = validateMatrixFiles(config_, errorMsg);
        if (fileCheckCode != ErrorCode::SUCCESS)
        {
            LOG_ERROR("[Undistortion] Weight file validation failed: %s", errorMsg.c_str());
            LOG_DEBUGF("validateMatrixFiles failed: %s", errorMsg.c_str()); // Log file check error
            return fileCheckCode;
        }
        // Load interpolation matrices
        try
        {
            LOG_INFO("[Undistortion] Start loading all interpolation matrices");
            loadAllMatricesMat();
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Undistortion] Exception while loading interpolation matrices: %s", e.what());
            initialized_ = false;
            return ErrorCode::FILE_ERROR;
        }
        initialized_ = true;
        LOG_INFO("[Undistortion] setConfig success");
        return ErrorCode::SUCCESS;
    }

    /**
     * @brief Check if configured
     */
    bool isConfigured() const noexcept
    {
        LOG_DEBUGF("ImageUndistorterImpl::isConfigured() - initialized_=%d", initialized_); // Log config state
        return initialized_;
    }

    /**
     * @brief Perform image undistortion (unified interface name: infer)
     */
    ErrorCode infer(const uint8_t *inputImage, int width, int height, uint8_t *outputImage, BaseResult &result) noexcept
    {
        LOG_INFO("[Undistortion] call infer");
        LOG_DEBUGF("infer - inputImage=%p, width=%d, height=%d, outputImage=%p", inputImage, width, height, outputImage); // Log input params
        result.reset();
        if (!initialized_)
        {
            LOG_ERROR("[Undistortion] Not initialized, cannot perform infer");
            LOG_DEBUG("UndistorterImpl not initialized, aborting infer"); // Log not initialized
            result.errorCode = ErrorCode::INVALID_CONFIGURATION;
            std::snprintf(result.errorMessage, sizeof(result.errorMessage), "Undistorter not initialized");
            return result.errorCode;
        }
        if (!inputImage || !outputImage || width <= 0 || height <= 0)
        {
            LOG_ERROR("[Undistortion] Invalid input/output image parameters");
            LOG_DEBUGF("infer invalid input: inputImage=%p, outputImage=%p, width=%d, height=%d", inputImage, outputImage, width, height); // Log invalid input
            result.errorCode = ErrorCode::INVALID_INPUT;
            std::snprintf(result.errorMessage, sizeof(result.errorMessage), "Invalid input/output image or size");
            return result.errorCode;
        }
        // Check if size matches matrix
        if (width != matrixWidth_ || height != matrixHeight_)
        {
            LOG_ERROR("[Undistortion] Input image size does not match matrix");
            LOG_DEBUGF("infer size mismatch: width=%d, matrixWidth_=%d, height=%d, matrixHeight_=%d", width, matrixWidth_, height, matrixHeight_); // Log size mismatch
            result.errorCode = ErrorCode::INVALID_INPUT;
            std::snprintf(result.errorMessage, sizeof(result.errorMessage), "Input image size does not match undistortion matrix");
            return result.errorCode;
        }
        auto start = std::chrono::high_resolution_clock::now();
        try
        {
            // Choose method: for-loop lookup table or OpenCV remap
            LOG_DEBUG("infer: preparing input/output matrices"); // Log matrix prep
            cv::Mat inputMat(height, width, (channels_ == 1) ? CV_8UC1 : CV_8UC3, const_cast<uint8_t *>(inputImage), inputStride_);
            cv::Mat outputMat = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
            forLoopUndistortMat(inputMat, outputMat);
            cv::Mat outMat(height, width, CV_8UC1, outputImage);
            // Write outputMat data to outputImage
            outputMat.convertTo(outMat, CV_8UC1);
            result.errorCode = ErrorCode::SUCCESS;
            LOG_INFO("[Undistortion] infer finished successfully"); // Log success
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[Undistortion] Exception during infer: %s", e.what());
            LOG_DEBUGF("infer exception: %s", e.what()); // Log exception
            result.errorCode = ErrorCode::PROCESSING_ERROR;
            std::snprintf(result.errorMessage, sizeof(result.errorMessage), "%s", e.what());
        }
        catch (...)
        {
            LOG_ERROR("[Undistortion] Unknown exception during infer, possibly memory out of bounds");
            LOG_DEBUG("infer unknown exception, possibly memory issue"); // Log unknown exception
            result.errorCode = ErrorCode::PROCESSING_ERROR;
            snprintf(result.errorMessage, sizeof(result.errorMessage), "Unknown exception, possibly memory out of bounds");
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<float>(end - start).count();
        // 原因分析：LOG_INFO 宏的格式化字符串 "%.3fs" 只会被正确处理为浮点数输出，
        // 前提是 LOG_INFO 宏内部实现支持 printf 风格格式化（如 vsnprintf），
        // 否则可能直接原样输出字符串（如 spdlog/流式实现），导致 %.3fs 不被替换为数值。
        // 解决方法：将耗时拼接为字符串传入，或用 %.3f 并确认日志实现支持。
        // 推荐写法如下（兼容 printf 风格和流式日志库）：
        LOG_INFO("[Undistortion] infer finished, time cost %.3f s", result.processingTime);
        // 如果日志库不支持 printf 格式化，可改为：
        // LOG_INFO(std::string("[Undistortion] infer finished, time cost ") + std::to_string(result.processingTime) + " s");
        return result.errorCode;
    }

private:
    UndistortionConfig config_;
    bool initialized_ = false;
    int matrixWidth_ = 0;
    int matrixHeight_ = 0;
    int channels_ = 1;
    int inputStride_ = 0;
    int outputStride_ = 0;
    // Mat-style weight matrices
    cv::Mat point_u_, point_v_, weights_;
    // 4 channel interpolation matrices // Removed
    // std::vector<std::vector<std::vector<float>>> interpU_, interpV_, interpA_; // Removed

    /**
     * @brief Load weight matrices using OpenCV Mat style
     */
    void loadAllMatricesMat()
    {
        LOG_INFO("[Undistortion] Load weight matrices using OpenCV Mat style");
        LOG_DEBUGF("loadAllMatricesMat - matrixPath=%s, matrixWidth_=%d, matrixHeight_=%d", config_.matrixPath, matrixWidth_, matrixHeight_); // Log matrix load params
        point_u_ = cv::Mat(matrixHeight_, matrixWidth_, CV_32FC4, cv::Scalar(0, 0, 0, 0));
        point_v_ = cv::Mat(matrixHeight_, matrixWidth_, CV_32FC4, cv::Scalar(0, 0, 0, 0));
        weights_ = cv::Mat(matrixHeight_, matrixWidth_, CV_32FC4, cv::Scalar(0, 0, 0, 0));
        // Load U
        std::vector<cv::Mat> uMats(4);
        for (int k = 0; k < 4; ++k)
        {
            std::string upath = std::string(config_.matrixPath) + "/U_" + std::to_string(k + 1) + ".bin";
            LOG_DEBUGF("Loading U matrix: %s", upath.c_str()); // Log U file
            uMats[k] = readfiletoMat(matrixHeight_, matrixWidth_, upath);
        }
        for (int i = 0; i < matrixHeight_; ++i)
        {
            for (int j = 0; j < matrixWidth_; ++j)
            {
                for (int k = 0; k < 4; ++k)
                {
                    point_u_.at<cv::Vec4f>(i, j)[k] = uMats[k].at<float>(i, j);
                }
            }
        }
        // Load V
        std::vector<cv::Mat> vMats(4);
        for (int k = 0; k < 4; ++k)
        {
            std::string vpath = std::string(config_.matrixPath) + "/V_" + std::to_string(k + 1) + ".bin";
            LOG_DEBUGF("Loading V matrix: %s", vpath.c_str()); // Log V file
            vMats[k] = readfiletoMat(matrixHeight_, matrixWidth_, vpath);
        }
        for (int i = 0; i < matrixHeight_; ++i)
        {
            for (int j = 0; j < matrixWidth_; ++j)
            {
                for (int k = 0; k < 4; ++k)
                {
                    point_v_.at<cv::Vec4f>(i, j)[k] = vMats[k].at<float>(i, j);
                }
            }
        }
        // Load A
        std::vector<cv::Mat> aMats(4);
        for (int k = 0; k < 4; ++k)
        {
            std::string apath = std::string(config_.matrixPath) + "/A" + std::to_string(k + 1) + ".bin";
            LOG_DEBUGF("Loading A matrix: %s", apath.c_str()); // Log A file
            aMats[k] = readfiletoMat(matrixHeight_, matrixWidth_, apath);
        }
        for (int i = 0; i < matrixHeight_; ++i)
        {
            for (int j = 0; j < matrixWidth_; ++j)
            {
                for (int k = 0; k < 4; ++k)
                {
                    weights_.at<cv::Vec4f>(i, j)[k] = aMats[k].at<float>(i, j);
                }
            }
        }
        LOG_INFO("[Undistortion] Weight matrices loaded (Mat style)");
    }

    /**
     * @brief for-loop undistortion using OpenCV Mat style
     */
    void forLoopUndistortMat(const cv::Mat &input, cv::Mat &output)
    {
        LOG_DEBUG("forLoopUndistortMat - begin"); // Log function entry
        for (int v = 0; v < output.rows; ++v)
        {
            for (int u = 0; u < output.cols; ++u)
            {
                cv::Vec4f U = point_u_.at<cv::Vec4f>(v, u);
                cv::Vec4f V = point_v_.at<cv::Vec4f>(v, u);
                cv::Vec4f w = weights_.at<cv::Vec4f>(v, u);

                // Check if all four points are in valid range ([1, cols) and [1, rows))
                bool valid =
                    (U[0] >= 1 && U[0] < input.cols && V[0] >= 1 && V[0] < input.rows) &&
                    (U[1] >= 1 && U[1] < input.cols && V[1] >= 1 && V[1] < input.rows) &&
                    (U[2] >= 1 && U[2] < input.cols && V[2] >= 1 && V[2] < input.rows) &&
                    (U[3] >= 1 && U[3] < input.cols && V[3] >= 1 && V[3] < input.rows);

                if (valid)
                {
                    float pixel =
                        static_cast<float>(input.at<uint8_t>(static_cast<int>(V[0]), static_cast<int>(U[0]))) * w[0] +
                        static_cast<float>(input.at<uint8_t>(static_cast<int>(V[1]), static_cast<int>(U[1]))) * w[1] +
                        static_cast<float>(input.at<uint8_t>(static_cast<int>(V[2]), static_cast<int>(U[2]))) * w[2] +
                        static_cast<float>(input.at<uint8_t>(static_cast<int>(V[3]), static_cast<int>(U[3]))) * w[3];
                    output.at<float>(v, u) = pixel;
                }
                else
                {
                    output.at<float>(v, u) = 0.0f;
                }
            }
        }
        LOG_DEBUG("forLoopUndistortMat - end"); // Log function exit
    }

    /**
     * @brief Read a single weight file as Mat
     */
    cv::Mat readfiletoMat(int rows, int cols, const std::string &filename)
    {
        LOG_DEBUGF("readfiletoMat - filename=%s, rows=%d, cols=%d", filename.c_str(), rows, cols); // Log file read params
        cv::Mat mat(rows, cols, CV_32F, cv::Scalar(0));
        std::ifstream inputFile(filename, std::ios::binary);
        if (!inputFile)
        {
            LOG_ERROR("[Undistortion] Cannot open weight file: %s", filename.c_str());
            return mat;
        }
        inputFile.seekg(0, std::ios::end);
        std::streamsize size = inputFile.tellg();
        inputFile.seekg(0, std::ios::beg);
        size_t numFloats = size / sizeof(float);
        std::vector<float> data(numFloats);
        inputFile.read(reinterpret_cast<char *>(data.data()), size);
        inputFile.close();
        for (int i = 0; i < cols; ++i)
        {
            for (int j = 0; j < rows; ++j)
            {
                mat.at<float>(j, i) = data[j + i * rows];
            }
        }
        LOG_DEBUGF("readfiletoMat - loaded file %s successfully", filename.c_str()); // Log file read success
        return mat;
    }

    /**
     * @brief Validate all weight files exist and have correct size
     */
    ErrorCode validateMatrixFiles(const UndistortionConfig &config, std::string &errorMsg)
    {
        LOG_DEBUG("validateMatrixFiles - begin"); // Log function entry
        static const char *files[] = {
            "U_1.bin", "U_2.bin", "U_3.bin", "U_4.bin",
            "V_1.bin", "V_2.bin", "V_3.bin", "V_4.bin",
            "A1.bin", "A2.bin", "A3.bin", "A4.bin"};
        size_t expectedSize = static_cast<size_t>(config.matrixWidth) * config.matrixHeight * sizeof(float);
        for (const char *fname : files)
        {
            std::string fpath = std::string(config.matrixPath) + "/" + fname;
            LOG_DEBUGF("Checking weight file: %s", fpath.c_str()); // Log file check
            std::ifstream fin(fpath, std::ios::binary | std::ios::ate);
            if (!fin)
            {
                LOG_ERROR("[Undistortion] Weight file does not exist: %s", fpath.c_str());
                char buf[512];
                snprintf(buf, sizeof(buf),
                         "Weight file does not exist: %s", fpath.c_str());
                errorMsg = buf;
                return ErrorCode::FILE_ERROR;
            }
            size_t fileSize = fin.tellg();
            if (fileSize != expectedSize)
            {
                LOG_ERROR("[Undistortion] Weight file size mismatch: %s, expected: %zu, actual: %zu", fpath.c_str(), expectedSize, fileSize);
                LOG_DEBUGF("File size mismatch: %s, expected: %zu, actual: %zu", fpath.c_str(), expectedSize, fileSize); // Log size mismatch
                char buf[512];
                snprintf(buf, sizeof(buf),
                         "Weight file size mismatch: %s, expected: %zu bytes (width %d × height %d × 4), actual: %zu bytes", fpath.c_str(), expectedSize, config.matrixWidth, config.matrixHeight, fileSize);
                errorMsg = buf;
                return ErrorCode::FILE_ERROR;
            }
            LOG_DEBUGF("Weight file validated: %s", fpath.c_str()); // Log file validated
        }
        LOG_INFO("[Undistortion] All weight files validated");
        LOG_DEBUG("validateMatrixFiles - end"); // Log function exit
        return ErrorCode::SUCCESS;
    }

    /**
     * @brief for-loop undistortion implementation (original solution)
     */
    // void forLoopUndistort(const uint8_t *inputImage, int width, int height, uint8_t *outputImage) // Removed
    // {
    //     LOG_INFO("[Undistortion] Using for-loop lookup table solution");
    //     for (int c = 0; c < channels_; ++c)
    //     {
    //         const uint8_t *inPtr = inputImage + c;
    //         uint8_t *outPtr = outputImage + c;
    //         for (int row = 0; row < height; ++row)
    //         {
    //             for (int col = 0; col < width; ++col)
    //             {
    //                 float pixel = 0.0f;
    //                 bool valid = false;
    //                 for (int k = 0; k < 4; ++k)
    //                 {
    //                     int u = static_cast<int>(interpU_[row][col][k]);
    //                     int v = static_cast<int>(interpV_[row][col][k]);
    //                     float w = interpA_[row][col][k];
    //                     if (u >= 0 && u < width && v >= 0 && v < height)
    //                     {
    //                         pixel += inPtr[v * inputStride_ + u * channels_] * w;
    //                         valid = true;
    //                     }
    //                 }
    //                 outPtr[row * outputStride_ + col * channels_] = valid ? static_cast<uint8_t>(std::clamp(pixel, 0.0f, 255.0f)) : 0;
    //             }
    //         }
    //     }
    // }

    /**
     * @brief OpenCV remap efficient undistortion implementation
     * Only use U_1/V_1 to generate map_x/map_y, support grayscale and color
     */
    // void remapUndistort(const uint8_t *inputImage, int width, int height, uint8_t *outputImage) // Removed
    // {
    //     LOG_INFO("[Undistortion] Using OpenCV remap solution");
    //     // 1. Build map_x/map_y
    //     cv::Mat map_x(height, width, CV_32FC1);
    //     cv::Mat map_y(height, width, CV_32FC1);
    //     for (int row = 0; row < height; ++row)
    //     {
    //         for (int col = 0; col < width; ++col)
    //         {
    //             map_x.at<float>(row, col) = interpU_[row][col][0]; // Only use U_1
    //             map_y.at<float>(row, col) = interpV_[row][col][0]; // Only use V_1
    //         }
    //     }
    //     // 2. Build input Mat
    //     int type = (channels_ == 1) ? CV_8UC1 : CV_8UC3;
    //     cv::Mat src(height, width, type, const_cast<uint8_t *>(inputImage), inputStride_);
    //     cv::Mat dst(height, width, type, outputImage, outputStride_);
    //     // 3. remap
    //     cv::remap(src, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    // }
};

// =============================
// ImageUndistorter External Interface Implementation
// =============================
ImageUndistorter::ImageUndistorter() : pImpl_(std::make_unique<ImageUndistorterImpl>()) {}
ImageUndistorter::~ImageUndistorter() = default;
ImageUndistorter::ImageUndistorter(ImageUndistorter &&) noexcept = default;
ImageUndistorter &ImageUndistorter::operator=(ImageUndistorter &&) noexcept = default;

ErrorCode ImageUndistorter::setConfig(const UndistortionConfig &config)
{
    return pImpl_->setConfig(config);
}
bool ImageUndistorter::isConfigured() const noexcept
{
    return pImpl_->isConfigured();
}
ErrorCode ImageUndistorter::infer(const uint8_t *inputImage, int width, int height, uint8_t *outputImage, BaseResult &result) noexcept
{
    return pImpl_->infer(inputImage, width, height, outputImage, result);
}