#include "undist_api.h"

#include "undist_engine.h"
#include "undist_tables.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>

// ===================================================================
// C ABI 实现：句柄 / 生命周期 / 参数校验 / 错误串
// 真正的计算在 undist::Engine（TRT）与 undist::build_tables（CPU）
// ===================================================================

struct undist_ctx {
    std::mutex mu;
    undist::Tables tables;
    std::unique_ptr<undist::Engine> engine;
    std::string error;
    undist_timing last{};
    undist_info info{};
    bool ready = false;
};

namespace {

void SetError(undist_ctx* ctx, const std::string& msg) {
    if (ctx) ctx->error = msg;
}

void FillInfo(undist_ctx* ctx) {
    undist::Engine& e = *ctx->engine;
    undist_info& info = ctx->info;
    const double t_table = info.init_table_ms;
    const double t_engine = info.init_engine_ms;
    const double t_total = info.init_total_ms;
    std::memset(&info, 0, sizeof(info));
    info.table_bytes = e.table_bytes();
    info.engine_bytes = e.engine_bytes();
    info.device_bytes = e.device_bytes();
    info.table_bytes_per_pixel = 12.0;
    info.init_table_ms = t_table;
    info.init_engine_ms = t_engine;
    info.init_total_ms = t_total;
    info.width = e.width();
    info.height = e.height();
    info.channels = 1;
    info.memory_mode = e.memory_mode();
    info.table_variant = undist::kTableVariantF;
    info.sm_count = e.sm_count();
    std::snprintf(info.gpu_name, sizeof(info.gpu_name), "%s", e.gpu_name());
    std::snprintf(info.tensorrt_version, sizeof(info.tensorrt_version), "%s", e.trt_version());
}

} // namespace

extern "C" {

int32_t UNDIST_CALL undist_create(const char* matrix_dir_utf8,
                                  int32_t width,
                                  int32_t height,
                                  const undist_options* options,
                                  const char* engine_cache_path_utf8,
                                  undist_ctx** out_ctx) {
    if (!out_ctx) return UNDIST_ERR_INVALID_ARG;
    *out_ctx = nullptr;
    if (!matrix_dir_utf8 || matrix_dir_utf8[0] == '\0' || width <= 0 || height <= 0) {
        return UNDIST_ERR_INVALID_ARG;
    }

    undist_ctx* ctx = new (std::nothrow) undist_ctx();
    if (!ctx) return UNDIST_ERR_OUT_OF_MEMORY;
    *out_ctx = ctx;   // 失败也回写：调用方可 get_error 后再 release

    undist_options opt{};
    opt.memory_mode = 1;
    opt.device_id = 0;
    opt.channels = 1;
    opt.log_level = 0;
    if (options) opt = *options;

    const int channels = (opt.channels == 0) ? 1 : opt.channels;
    if (channels != 1) {
        SetError(ctx, "只支持单通道（channels=1）去畸变");
        return UNDIST_ERR_UNSUPPORTED;
    }

    std::lock_guard<std::mutex> lock(ctx->mu);
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    int code = undist::build_tables(matrix_dir_utf8, width, height, ctx->tables, err,
                                    opt.log_level);
    const auto t1 = std::chrono::steady_clock::now();
    ctx->info.init_table_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (code != UNDIST_OK) {
        SetError(ctx, err.empty() ? "矩阵加载失败" : err);
        return code;
    }

    ctx->engine = std::make_unique<undist::Engine>();
    const std::string cache = engine_cache_path_utf8 ? engine_cache_path_utf8 : "";
    code = ctx->engine->initialize(ctx->tables, opt.memory_mode, cache, opt.device_id,
                                   opt.log_level, err);
    const auto t2 = std::chrono::steady_clock::now();
    ctx->info.init_engine_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    ctx->info.init_total_ms =
        std::chrono::duration<double, std::milli>(t2 - t0).count();
    if (code != UNDIST_OK) {
        SetError(ctx, err.empty() ? "引擎初始化失败" : err);
        return code;
    }

    FillInfo(ctx);
    ctx->ready = true;
    return UNDIST_OK;
}

int32_t UNDIST_CALL undist_process(undist_ctx* ctx,
                                   const uint8_t* src,
                                   int32_t src_stride,
                                   uint8_t* dst,
                                   int32_t dst_stride,
                                   int32_t host_flags,
                                   undist_timing* out_timing) {
    if (!ctx) return UNDIST_ERR_INVALID_HANDLE;
    std::lock_guard<std::mutex> lock(ctx->mu);
    if (!ctx->ready) {
        SetError(ctx, "引擎未就绪");
        return UNDIST_ERR_NOT_READY;
    }
    if (!src || !dst) {
        SetError(ctx, "src/dst 指针为空");
        return UNDIST_ERR_INVALID_IMAGE;
    }
    if (src_stride < ctx->info.width || dst_stride < ctx->info.width) {
        SetError(ctx, "stride 小于图像宽度");
        return UNDIST_ERR_INVALID_IMAGE;
    }
    if (host_flags != 0 && host_flags != 1) {
        SetError(ctx, "host_flags 仅支持 0 或 1");
        return UNDIST_ERR_INVALID_ARG;
    }

    std::string err;
    undist_timing timing{};
    const int code = ctx->engine->infer(src, src_stride, dst, dst_stride,
                                        host_flags == 1, timing, err);
    if (code != UNDIST_OK) {
        SetError(ctx, err.empty() ? "推理失败" : err);
        return code;
    }
    ctx->last = timing;
    if (out_timing) *out_timing = timing;
    return UNDIST_OK;
}

int32_t UNDIST_CALL undist_bench_kernel(undist_ctx* ctx,
                                        const uint8_t* src,
                                        int32_t iterations,
                                        undist_timing* out_timing) {
    if (!ctx) return UNDIST_ERR_INVALID_HANDLE;
    if (!out_timing) return UNDIST_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->mu);
    if (!ctx->ready) {
        SetError(ctx, "引擎未就绪");
        return UNDIST_ERR_NOT_READY;
    }
    if (!src || iterations < 1) {
        SetError(ctx, "bench_kernel 参数非法");
        return UNDIST_ERR_INVALID_ARG;
    }
    std::string err;
    undist_timing timing{};
    const int code = ctx->engine->bench_kernel(src, iterations, timing, err);
    if (code != UNDIST_OK) {
        SetError(ctx, err.empty() ? "bench 失败" : err);
        return code;
    }
    *out_timing = timing;
    return UNDIST_OK;
}

int32_t UNDIST_CALL undist_get_info(undist_ctx* ctx, undist_info* out_info) {
    if (!ctx) return UNDIST_ERR_INVALID_HANDLE;
    if (!out_info) return UNDIST_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->mu);
    *out_info = ctx->info;
    return UNDIST_OK;
}

int32_t UNDIST_CALL undist_get_last_timing(undist_ctx* ctx, undist_timing* out_timing) {
    if (!ctx) return UNDIST_ERR_INVALID_HANDLE;
    if (!out_timing) return UNDIST_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->mu);
    *out_timing = ctx->last;
    return UNDIST_OK;
}

int32_t UNDIST_CALL undist_set_memory_mode(undist_ctx* ctx, int32_t memory_mode) {
    if (!ctx) return UNDIST_ERR_INVALID_HANDLE;
    if (memory_mode != 0 && memory_mode != 1) return UNDIST_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->mu);
    if (!ctx->ready) return UNDIST_ERR_NOT_READY;
    std::string err;
    const int code = ctx->engine->set_memory_mode(memory_mode, err);
    if (code != UNDIST_OK) {
        SetError(ctx, err.empty() ? "切换内存模式失败" : err);
        return code;
    }
    ctx->info.memory_mode = ctx->engine->memory_mode();
    return UNDIST_OK;
}

const char* UNDIST_CALL undist_get_error(undist_ctx* ctx) {
    if (!ctx) return "undist_ctx is null";
    return ctx->error.c_str();
}

int32_t UNDIST_CALL undist_host_register(void* ptr, int64_t bytes) {
    if (!ptr || bytes <= 0) return UNDIST_ERR_INVALID_ARG;
    const cudaError_t e = cudaHostRegister(ptr, static_cast<size_t>(bytes),
                                           cudaHostRegisterDefault);
    return e == cudaSuccess ? UNDIST_OK : UNDIST_ERR_INIT_FAILED;
}

void UNDIST_CALL undist_host_unregister(void* ptr) {
    if (ptr) cudaHostUnregister(ptr);
}

void UNDIST_CALL undist_release(undist_ctx* ctx) {
    delete ctx;
}

} // extern "C"
