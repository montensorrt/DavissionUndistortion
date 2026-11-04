#pragma once

/**
 * @file undist_api.h
 * @brief 去畸变（Undistortion）TensorRT 推理 C-API —— 供 C++ / C# P/Invoke 调用
 *
 * 契约（与 INFER/deploy/trt/include/*.h 同构）：
 *   - 初始化：传入<b>去畸变矩阵目录</b>（U_1..U_4 / V_1..V_4 / A1..A4 共 12 个 .bin）
 *             与目标图尺寸；库内部完成「矩阵加载 → 12B/px 紧凑表 → TRT 引擎构建 → 表上显存」。
 *   - 推理：输入为 H×W 单通道 uint8 连续内存（可带行跨距 stride），输出写入<b>调用方提供</b>的
 *           同尺寸缓冲。全程只在内存中，不做任何落盘（引擎缓存是可选的能力，默认关闭）。
 *   - 线程：一个句柄不可并发使用；C-API 内部有互斥锁串行化（防御性，不做并行承诺）。
 *   - 生命周期：undist_create 失败时仍可能回写句柄，调用方应先 undist_get_error 再 release。
 *
 * 数值语义（与 CPU 版本严格对齐）：
 *   out = floor(clamp(Σ_{k=0..3} src[idx_k] * w_k, 0, 255) + 0.5)
 *   其中 idx_k 由「U/V 四抽头全部在界内」的与条件给出，越界像素读取值为 0 的哑元；
 *   权重由 fu/fv 重建：w0=(1-fu)(1-fv), w1=fu(1-fv), w2=(1-fu)fv, w3=fu*fv。
 */

#include <stdint.h>

#ifdef _WIN32
  #ifdef UNDIST_API_EXPORTS
    #define UNDIST_API __declspec(dllexport)
  #else
    #define UNDIST_API __declspec(dllimport)
  #endif
  #define UNDIST_CALL __cdecl
#else
  #define UNDIST_API __attribute__((visibility("default")))
  #define UNDIST_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** 错误码（与 C# UndistErrorCode 一一对应；0 成功，负值失败） */
typedef enum {
    UNDIST_OK                    =  0,
    UNDIST_ERR_INIT_FAILED       = -1,   /**< CUDA / TensorRT 初始化失败 */
    UNDIST_ERR_INVALID_HANDLE    = -2,   /**< 句柄为空 / 已释放 / 未就绪 */
    UNDIST_ERR_INVALID_IMAGE     = -3,   /**< 指针为空 / 宽高与初始化不符 / stride < width */
    UNDIST_ERR_INFERENCE_FAILED  = -4,   /**< enqueueV3 失败或取不到输出张量 */
    UNDIST_ERR_NOT_READY         = -5,   /**< 表 / 引擎尚未就绪 */
    UNDIST_ERR_MATRIX_MISSING    = -6,   /**< 矩阵目录或 12 个 .bin 缺失 */
    UNDIST_ERR_MATRIX_SIZE       = -7,   /**< .bin 文件大小与 width*height*4 不符 */
    UNDIST_ERR_OUT_OF_MEMORY     = -8,   /**< 主机 / 显存分配失败 */
    UNDIST_ERR_ENGINE_BUILD      = -9,   /**< TRT 引擎构建或反序列化失败 */
    UNDIST_ERR_UNSUPPORTED       = -10,  /**< 通道数不是 1 等未实现能力 */
    UNDIST_ERR_INVALID_ARG       = -11   /**< 出参为空 / 参数自相矛盾 */
} undist_error_code;

/** 初始化选项（全部为 32 位字段，保证 C/C# 结构体 blittable） */
typedef struct {
    int32_t memory_mode;   /**< 0 = pageable（对照用）, 1 = pinned（工程推荐，默认） */
    int32_t device_id;     /**< CUDA 设备号，默认 0 */
    int32_t channels;      /**< 仅支持 1；0 等同 1 */
    int32_t log_level;     /**< 0 = 静默, 1 = 关键节点, 2 = 逐阶段明细 */
} undist_options;

/** 运行期信息（C# 不解析任何配置文件，参数一律由此回传） */
typedef struct {
    int64_t table_bytes;           /**< 常驻显存的索引/权重表字节数（12 B/px） */
    int64_t engine_bytes;          /**< 序列化引擎字节数（未缓存时为内存中大小） */
    int64_t device_bytes;          /**< 显存总占用（表 + 输入/输出） */
    double  table_bytes_per_pixel; /**< = table_bytes / (width*height)，应为 12.0 */
    double  init_table_ms;         /**< 矩阵加载 + 紧凑表构建耗时 */
    double  init_engine_ms;        /**< 引擎构建/加载 + 表上显存 + 暂存缓冲耗时 */
    double  init_total_ms;         /**< undist_create 总耗时 */
    int32_t width;
    int32_t height;
    int32_t channels;
    int32_t memory_mode;           /**< 当前生效的搬运模式 */
    int32_t table_variant;         /**< 1 = 变体 F（idx0 + fu + fv） */
    int32_t sm_count;              /**< GPU SM 数 */
    char    gpu_name[96];
    char    tensorrt_version[32];
} undist_info;

/** 分阶段耗时（毫秒；串行口径：每段结束都做一次流同步） */
typedef struct {
    double h2d_ms;    /**< 主机 → 设备（pinned 时不含行拷贝；pageable 含驱动中转） */
    double infer_ms;  /**< enqueueV3 + cudaStreamSynchronize（纯 kernel 墙钟） */
    double d2h_ms;    /**< 设备 → 主机 */
    double total_ms;  /**< 整个 undist_process 调用（含主机行拷贝/重排） */
} undist_timing;

/** 不透明句柄 */
typedef struct undist_ctx undist_ctx;

/**
 * @brief 由矩阵目录初始化
 *
 * @param matrix_dir_utf8     去畸变矩阵目录（UTF-8），内含 U_1..U_4 / V_1..V_4 / A1..A4
 * @param width               目标图宽度（= 矩阵宽度）
 * @param height              目标图高度（= 矩阵高度）
 * @param options             可为 NULL（等价 memory_mode=1 / device=0 / channels=1 / log=0）
 * @param engine_cache_path   可为 NULL；非空时「存在则反序列化，不存在则构建后写盘」
 * @param out_ctx             输出句柄；<b>失败时也可能非空</b>，调用方应先取错误再 release
 *
 * @return 0 成功；-6 矩阵缺失 / -7 大小不符 / -9 引擎失败 / -8 显存不足 / -1 初始化失败
 */
UNDIST_API int32_t UNDIST_CALL undist_create(const char* matrix_dir_utf8,
                                             int32_t width,
                                             int32_t height,
                                             const undist_options* options,
                                             const char* engine_cache_path_utf8,
                                             undist_ctx** out_ctx);

/**
 * @brief 单图去畸变（内存进 → 内存出，零落盘）
 *
 * @param ctx         句柄
 * @param src         输入 H×W 单通道 uint8
 * @param src_stride  输入行跨距（字节），>= width
 * @param dst         输出 H×W 单通道 uint8（调用方分配）
 * @param dst_stride  输出行跨距（字节），>= width
 * @param host_flags  主机缓冲语义位掩码：
 *                      bit0 = 1 表示 src/dst 已是<b>页锁内存</b>（调用方用 cudaHostAlloc /
 *                             GC.AllocateArray(pinned:true) 等分配），库将跳过内部暂存、
 *                             直接做异步 H2D/D2H —— 工程推荐路径；
 *                      bit0 = 0 表示普通 pageable 内存，库内部按 memory_mode 处理。
 * @param out_timing  可为 NULL；非 NULL 时回填分段耗时
 *
 * @return 0 成功；-3 参数非法；-4 推理失败；-2 句柄无效
 */
UNDIST_API int32_t UNDIST_CALL undist_process(undist_ctx* ctx,
                                              const uint8_t* src,
                                              int32_t src_stride,
                                              uint8_t* dst,
                                              int32_t dst_stride,
                                              int32_t host_flags,
                                              undist_timing* out_timing);

/**
 * @brief 纯 kernel 基准：图像只搬运一次，然后连续 enqueue iterations 次
 *
 * 用于把「表带宽受限的 kernel 时间」与「PCIe 搬运」解耦（对应报告里的 GPU-G4）。
 *
 * @param ctx          句柄
 * @param src          输入 H×W 单通道 uint8（pageable 亦可）
 * @param iterations   迭代次数，>= 1
 * @param out_timing   infer_ms = 平均 kernel 墙钟；h2d/d2h = 0；total_ms = 上述平均值
 */
UNDIST_API int32_t UNDIST_CALL undist_bench_kernel(undist_ctx* ctx,
                                                   const uint8_t* src,
                                                   int32_t iterations,
                                                   undist_timing* out_timing);

/** 查询运行期信息 */
UNDIST_API int32_t UNDIST_CALL undist_get_info(undist_ctx* ctx, undist_info* out_info);

/** 查询最近一次推理的分段耗时（与 undist_process 回填值一致） */
UNDIST_API int32_t UNDIST_CALL undist_get_last_timing(undist_ctx* ctx, undist_timing* out_timing);

/**
 * @brief 切换主机搬运模式（0 = pageable / 1 = pinned）
 *
 * 仅重建主机侧暂存缓冲，<b>不重建引擎、不重传表</b>；用于同口径 A/B 对照。
 */
UNDIST_API int32_t UNDIST_CALL undist_set_memory_mode(undist_ctx* ctx, int32_t memory_mode);

/** 最近一次错误描述（UTF-8，生命周期同句柄，不可 free；句柄为空时返回静态串） */
UNDIST_API const char* UNDIST_CALL undist_get_error(undist_ctx* ctx);

/**
 * @brief 把调用方已有的主机缓冲注册为 CUDA 页锁内存（供 C# 等托管语言使用）
 *
 * .NET 的 <c>GCHandle(Pinned)</c> / POH 只保证「地址不被 GC 移动」，并不是操作系统的
 * 非分页内存；必须经 cudaHostRegister 后，host_flags=1 才是真正的零拷贝异步 DMA。
 *
 * @param ptr    主机地址（需页对齐，GCHandle/POH 满足）
 * @param bytes  长度
 * @return 0 成功；-1 注册失败（页锁内存不足 / 地址非法）；-11 参数非法
 */
UNDIST_API int32_t UNDIST_CALL undist_host_register(void* ptr, int64_t bytes);

/** 解除页锁注册（与 undist_host_register 配对；NULL 安全） */
UNDIST_API void UNDIST_CALL undist_host_unregister(void* ptr);

/** 释放句柄（幂等：重复调用 / NULL 安全） */
UNDIST_API void UNDIST_CALL undist_release(undist_ctx* ctx);

#ifdef __cplusplus
}
#endif
