#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "undist_api.h"
#include "undist_tables.h"

namespace undist {

// ===================================================================
// TRT 引擎层：表上传、引擎构建/缓存、单图推理、纯 kernel 基准
// 对上层隐藏 TensorRT / CUDA 头文件（Pimpl）
// ===================================================================
class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /**
     * @brief 初始化：CUDA 上下文 → 引擎（缓存/构建）→ 表上显存 → 主机暂存缓冲
     * @return undist_error_code
     */
    int initialize(const Tables& tables,
                   int memory_mode,
                   const std::string& engine_cache_path_utf8,
                   int device_id,
                   int log_level,
                   std::string& err);

    int infer(const uint8_t* src, int src_stride,
              uint8_t* dst, int dst_stride,
              bool host_pinned,
              undist_timing& timing, std::string& err);

    int bench_kernel(const uint8_t* src, int iterations,
                     undist_timing& timing, std::string& err);

    int set_memory_mode(int memory_mode, std::string& err);

    int memory_mode() const;
    int width() const;
    int height() const;
    int64_t table_bytes() const;
    int64_t engine_bytes() const;
    int64_t device_bytes() const;
    const char* gpu_name() const;
    int sm_count() const;
    const char* trt_version() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace undist
