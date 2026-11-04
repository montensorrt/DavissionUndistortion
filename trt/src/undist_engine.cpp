#include "undist_engine.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>

namespace fs = std::filesystem;

namespace undist {
namespace {

using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void LogMsg(int log_level, int need, const char* fmt, ...) {
    if (log_level < need) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[undist][engine] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}

struct TrtDeleter {
    template <typename T>
    void operator()(T* p) const { delete p; }
};
template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter>;

class TrtLogger : public nvinfer1::ILogger {
public:
    int level = 0;
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kERROR) {
            std::fprintf(stderr, "[undist][trt][E] %s\n", msg);
        } else if (sev == Severity::kWARNING && level >= 1) {
            std::fprintf(stderr, "[undist][trt][W] %s\n", msg);
        } else if (level >= 2) {
            std::fprintf(stderr, "[undist][trt][I] %s\n", msg);
        }
    }
};

#pragma pack(push, 1)
struct CacheHeader {
    char magic[4];        // "UDV1"
    int32_t version;      // 1
    int32_t width;
    int32_t height;
    int32_t variant;      // 1 = F
    int32_t trt_version;  // ::getInferLibVersion()
    int64_t table_bytes;
};
#pragma pack(pop)
static_assert(sizeof(CacheHeader) == 32, "cache header must be 32 bytes");

int CudaCheck(cudaError_t e, const char* what, std::string& err, int code) {
    if (e == cudaSuccess) return UNDIST_OK;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s 失败: %s (%d)", what, cudaGetErrorString(e), int(e));
    err = buf;
    if (e == cudaErrorMemoryAllocation || e == cudaErrorDevicesUnavailable) {
        return UNDIST_ERR_OUT_OF_MEMORY;
    }
    return code;
}

void CopyRows(const uint8_t* src, int src_stride, uint8_t* dst, int dst_stride,
              int width, int height) {
    if (src_stride == width && dst_stride == width) {
        std::memcpy(dst, src, static_cast<size_t>(width) * height);
        return;
    }
    for (int y = 0; y < height; ++y) {
        std::memcpy(dst + static_cast<size_t>(y) * dst_stride,
                    src + static_cast<size_t>(y) * src_stride, width);
    }
}

} // namespace

// ===================================================================
// Impl
// ===================================================================
struct Engine::Impl {
    int width = 0;
    int height = 0;
    int64_t pixels = 0;
    int64_t pad = 0;             // 平坦输入尾部补零长度（覆盖哑元索引）
    int memory_mode = 1;
    int device_id = 0;
    int log_level = 0;
    int64_t engine_bytes = 0;
    int64_t table_bytes = 0;

    TrtLogger logger;
    TrtPtr<nvinfer1::IBuilder> builder;
    TrtPtr<nvinfer1::IRuntime> runtime;
    TrtPtr<nvinfer1::ICudaEngine> engine;
    TrtPtr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream = nullptr;
    int32_t* d_idx = nullptr;
    float* d_fu = nullptr;
    float* d_fv = nullptr;
    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    uint8_t* h_in = nullptr;   // pinned staging (nullptr = pageable)
    uint8_t* h_out = nullptr;

    char gpu_name[96] = {0};
    int sm_count = 0;
    char trt_version[32] = {0};

    ~Impl() { release_cuda(); }

    void release_cuda() noexcept {
        if (device_id >= 0) cudaSetDevice(device_id);
        if (h_in) { cudaFreeHost(h_in); h_in = nullptr; }
        if (h_out) { cudaFreeHost(h_out); h_out = nullptr; }
        if (d_idx) { cudaFree(d_idx); d_idx = nullptr; }
        if (d_fu) { cudaFree(d_fu); d_fu = nullptr; }
        if (d_fv) { cudaFree(d_fv); d_fv = nullptr; }
        if (d_in) { cudaFree(d_in); d_in = nullptr; }
        if (d_out) { cudaFree(d_out); d_out = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
        // TRT 对象在 CUDA 释放之后销毁（顺序由 unique_ptr 的显式 reset 保证）
        context.reset();
        engine.reset();
        runtime.reset();
        builder.reset();
    }

    int allocate_host_staging(std::string& err) {
        if (memory_mode != 1) return UNDIST_OK;
        if (h_in && h_out) return UNDIST_OK;
        const size_t n = static_cast<size_t>(pixels);
        cudaError_t e = cudaHostAlloc(reinterpret_cast<void**>(&h_in), n, cudaHostAllocDefault);
        if (e != cudaSuccess) {
            h_in = nullptr;
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                          "页锁内存分配失败(%zu MB): %s；已回退 pageable 模式",
                          n / (1024 * 1024), cudaGetErrorString(e));
            LogMsg(log_level, 0, "%s", buf);
            memory_mode = 0;
            return UNDIST_OK;
        }
        e = cudaHostAlloc(reinterpret_cast<void**>(&h_out), n, cudaHostAllocDefault);
        if (e != cudaSuccess) {
            cudaFreeHost(h_in);
            h_in = nullptr;
            memory_mode = 0;
            LogMsg(log_level, 0, "页锁内存分配失败(输出): %s；已回退 pageable 模式",
                   cudaGetErrorString(e));
            return UNDIST_OK;
        }
        LogMsg(log_level, 1, "pinned staging allocated: 2 x %.1f MB", double(n) / (1024.0 * 1024.0));
        return UNDIST_OK;
    }

    // ---------------- 网络构建（20 层，纯原生算子，无插件） ----------------
    TrtPtr<nvinfer1::ICudaEngine> build_network(int log_level, std::string& err) {
        const int64_t N = pixels;
        const int32_t H = height;
        const int32_t W = width;

        builder.reset(nvinfer1::createInferBuilder(logger));
        if (!builder) {
            err = "createInferBuilder 返回空";
            return nullptr;
        }
        const uint32_t flags = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        TrtPtr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(flags));
        if (!net) {
            err = "createNetworkV2 返回空";
            return nullptr;
        }

        auto add_input = [&](const char* name, nvinfer1::DataType dt, nvinfer1::Dims dims) {
            return net->addInput(name, dt, dims);
        };
        nvinfer1::ITensor* img = add_input("img", nvinfer1::DataType::kUINT8,
                                           nvinfer1::Dims4{1, 1, H, W});
        nvinfer1::ITensor* idx0 = add_input("idx0", nvinfer1::DataType::kINT32,
                                            nvinfer1::Dims{2, {1, static_cast<int32_t>(N)}});
        nvinfer1::ITensor* fu = add_input("fu", nvinfer1::DataType::kFLOAT,
                                          nvinfer1::Dims{2, {1, static_cast<int32_t>(N)}});
        nvinfer1::ITensor* fv = add_input("fv", nvinfer1::DataType::kFLOAT,
                                          nvinfer1::Dims{2, {1, static_cast<int32_t>(N)}});
        if (!img || !idx0 || !fu || !fv) {
            err = "添加网络输入失败";
            return nullptr;
        }

        nvinfer1::ILayer* cast = net->addCast(*img, nvinfer1::DataType::kFLOAT);
        if (!cast) { err = "addCast 失败"; return nullptr; }
        nvinfer1::IShuffleLayer* flat_img = net->addShuffle(*cast->getOutput(0));
        if (!flat_img) { err = "addShuffle 失败"; return nullptr; }
        flat_img->setReshapeDimensions(nvinfer1::Dims{2, {1, static_cast<int32_t>(N)}});

        // 平坦输入尾部 +PAD 个 0，让越界索引（= N）及其 +1/+W/+W+1 派生索引读到 0
        std::vector<float> pad_zeros(static_cast<size_t>(pad), 0.0f);
        nvinfer1::IConstantLayer* zeros = net->addConstant(
            nvinfer1::Dims{2, {1, static_cast<int32_t>(pad)}},
            nvinfer1::Weights{nvinfer1::DataType::kFLOAT, pad_zeros.data(), pad});
        if (!zeros) { err = "addConstant(pad) 失败"; return nullptr; }

        nvinfer1::ITensor* cat_in[2] = {flat_img->getOutput(0), zeros->getOutput(0)};
        nvinfer1::IConcatenationLayer* cat = net->addConcatenation(cat_in, 2);
        if (!cat) { err = "addConcatenation 失败"; return nullptr; }
        cat->setAxis(1);

        // idx0 + 1 / + W / + W+1：用 ElementWise(SUM) 派生，避免额外 3 张 int32 表
        int32_t one_val = 1;
        int32_t w_val = W;
        nvinfer1::IConstantLayer* c_one = net->addConstant(
            nvinfer1::Dims{2, {1, 1}},
            nvinfer1::Weights{nvinfer1::DataType::kINT32, &one_val, 1});
        nvinfer1::IConstantLayer* c_w = net->addConstant(
            nvinfer1::Dims{2, {1, 1}},
            nvinfer1::Weights{nvinfer1::DataType::kINT32, &w_val, 1});
        if (!c_one || !c_w) { err = "addConstant(index) 失败"; return nullptr; }

        nvinfer1::IElementWiseLayer* d1 = net->addElementWise(
            *idx0, *c_one->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        nvinfer1::IElementWiseLayer* d2 = net->addElementWise(
            *idx0, *c_w->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        if (!d1 || !d2) { err = "addElementWise(index) 失败"; return nullptr; }
        nvinfer1::IElementWiseLayer* d3 = net->addElementWise(
            *d1->getOutput(0), *c_w->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        if (!d3) { err = "addElementWise(index) 失败"; return nullptr; }

        nvinfer1::ITensor* idx_tensors[4] = {
            idx0, d1->getOutput(0), d2->getOutput(0), d3->getOutput(0)};
        nvinfer1::ITensor* gathered[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int k = 0; k < 4; ++k) {
            nvinfer1::IGatherLayer* g = net->addGather(*cat->getOutput(0), *idx_tensors[k], 1);
            if (!g) { err = "addGather 失败"; return nullptr; }
            g->setMode(nvinfer1::GatherMode::kELEMENT);
            gathered[k] = g->getOutput(0);
        }

        // 权重重建：w0=(1-fu)(1-fv) w1=fu(1-fv) w2=(1-fu)fv w3=fu*fv
        float one_f = 1.0f;
        nvinfer1::IConstantLayer* c_onef = net->addConstant(
            nvinfer1::Dims{2, {1, 1}},
            nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &one_f, 1});
        if (!c_onef) { err = "addConstant(1.0f) 失败"; return nullptr; }
        nvinfer1::IElementWiseLayer* omfu = net->addElementWise(
            *c_onef->getOutput(0), *fu, nvinfer1::ElementWiseOperation::kSUB);
        nvinfer1::IElementWiseLayer* omfv = net->addElementWise(
            *c_onef->getOutput(0), *fv, nvinfer1::ElementWiseOperation::kSUB);
        if (!omfu || !omfv) { err = "addElementWise(1-f) 失败"; return nullptr; }

        nvinfer1::ITensor* weights[4] = {nullptr, nullptr, nullptr, nullptr};
        nvinfer1::IElementWiseLayer* w0 = net->addElementWise(
            *omfu->getOutput(0), *omfv->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
        nvinfer1::IElementWiseLayer* w1 = net->addElementWise(
            *fu, *omfv->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
        nvinfer1::IElementWiseLayer* w2 = net->addElementWise(
            *omfu->getOutput(0), *fv, nvinfer1::ElementWiseOperation::kPROD);
        nvinfer1::IElementWiseLayer* w3 = net->addElementWise(
            *fu, *fv, nvinfer1::ElementWiseOperation::kPROD);
        if (!w0 || !w1 || !w2 || !w3) { err = "addElementWise(weight) 失败"; return nullptr; }
        weights[0] = w0->getOutput(0);
        weights[1] = w1->getOutput(0);
        weights[2] = w2->getOutput(0);
        weights[3] = w3->getOutput(0);

        nvinfer1::ITensor* acc = nullptr;
        for (int k = 0; k < 4; ++k) {
            nvinfer1::IElementWiseLayer* mul = net->addElementWise(
                *gathered[k], *weights[k], nvinfer1::ElementWiseOperation::kPROD);
            if (!mul) { err = "addElementWise(mul) 失败"; return nullptr; }
            if (k == 0) {
                acc = mul->getOutput(0);
            } else {
                nvinfer1::IElementWiseLayer* sum = net->addElementWise(
                    *acc, *mul->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
                if (!sum) { err = "addElementWise(acc) 失败"; return nullptr; }
                acc = sum->getOutput(0);
            }
        }

        nvinfer1::IActivationLayer* clip = net->addActivation(
            *acc, nvinfer1::ActivationType::kCLIP);
        if (!clip) { err = "addActivation(CLIP) 失败"; return nullptr; }
        clip->setAlpha(0.0f);
        clip->setBeta(255.0f);

        float half = 0.5f;
        nvinfer1::IConstantLayer* c_half = net->addConstant(
            nvinfer1::Dims{2, {1, 1}},
            nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &half, 1});
        nvinfer1::IElementWiseLayer* add_half = net->addElementWise(
            *clip->getOutput(0), *c_half->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
        nvinfer1::IUnaryLayer* flo = net->addUnary(
            *add_half->getOutput(0), nvinfer1::UnaryOperation::kFLOOR);
        nvinfer1::ICastLayer* out_cast = net->addCast(
            *flo->getOutput(0), nvinfer1::DataType::kUINT8);
        if (!c_half || !add_half || !flo || !out_cast) {
            err = "输出量化层构建失败";
            return nullptr;
        }
        nvinfer1::ITensor* out = out_cast->getOutput(0);
        out->setName("out_u8");
        net->markOutput(*out);

        TrtPtr<nvinfer1::IBuilderConfig> cfg(builder->createBuilderConfig());
        if (!cfg) { err = "createBuilderConfig 返回空"; return nullptr; }
        cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(256) << 20);
        cfg->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kNONE);

        const auto t0 = Clock::now();
        TrtPtr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*net, *cfg));
        if (!serialized) {
            err = "buildSerializedNetwork 返回空（详见 TRT 日志）";
            return nullptr;
        }
        LogMsg(log_level, 1, "engine built in %.1f ms, serialized=%.1f KB",
               ElapsedMs(t0, Clock::now()), double(serialized->size()) / 1024.0);

        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) { err = "createInferRuntime 返回空"; return nullptr; }
        nvinfer1::ICudaEngine* eng = runtime->deserializeCudaEngine(
            serialized->data(), serialized->size());
        if (!eng) { err = "deserializeCudaEngine 返回空"; return nullptr; }

        // 缓存序列化引擎（可选能力，默认关闭；不含任何图像数据）
        if (!cache_path.empty()) {
            std::error_code ec;
            fs::create_directories(cache_path.parent_path(), ec);
            std::ofstream fout(cache_path, std::ios::binary | std::ios::trunc);
            if (fout) {
                CacheHeader hdr{};
                std::memcpy(hdr.magic, "UDV1", 4);
                hdr.version = 1;
                hdr.width = width;
                hdr.height = height;
                hdr.variant = kTableVariantF;
                hdr.trt_version = ::getInferLibVersion();
                hdr.table_bytes = table_bytes;
                fout.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
                fout.write(reinterpret_cast<const char*>(serialized->data()),
                           static_cast<std::streamsize>(serialized->size()));
                LogMsg(log_level, 1, "engine cached -> %s", cache_path.u8string().c_str());
            } else {
                LogMsg(log_level, 0, "引擎缓存写入失败（忽略）: %s", cache_path.u8string().c_str());
            }
        }
        engine_bytes = static_cast<int64_t>(serialized->size());
        return TrtPtr<nvinfer1::ICudaEngine>(eng);
    }

    fs::path cache_path;

    TrtPtr<nvinfer1::ICudaEngine> load_or_build(int log_level, std::string& err) {
        if (cache_path.empty()) return build_network(log_level, err);

        std::ifstream fin(cache_path, std::ios::binary);
        if (fin) {
            CacheHeader hdr{};
            fin.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
            const bool ok = fin && std::memcmp(hdr.magic, "UDV1", 4) == 0 &&
                            hdr.version == 1 && hdr.width == width && hdr.height == height &&
                            hdr.variant == kTableVariantF &&
                            hdr.trt_version == ::getInferLibVersion() &&
                            hdr.table_bytes == table_bytes;
            if (ok) {
                fin.seekg(0, std::ios::end);
                const std::streamoff total = fin.tellg();
                const std::streamoff body = total - static_cast<std::streamoff>(sizeof(hdr));
                if (body > 0) {
                    std::vector<char> blob(static_cast<size_t>(body));
                    fin.seekg(sizeof(hdr), std::ios::beg);
                    fin.read(blob.data(), body);
                    if (fin) {
                        runtime.reset(nvinfer1::createInferRuntime(logger));
                        if (runtime) {
                            nvinfer1::ICudaEngine* eng = runtime->deserializeCudaEngine(
                                blob.data(), blob.size());
                            if (eng) {
                                engine_bytes = static_cast<int64_t>(blob.size());
                                LogMsg(log_level, 1, "engine loaded from cache: %s (%.1f KB)",
                                       cache_path.u8string().c_str(), double(blob.size()) / 1024.0);
                                return TrtPtr<nvinfer1::ICudaEngine>(eng);
                            }
                        }
                        LogMsg(log_level, 0, "引擎缓存反序列化失败，回退构建");
                    }
                }
            } else {
                LogMsg(log_level, 0, "引擎缓存头不匹配（尺寸/版本/表结构），回退构建");
            }
        }
        return build_network(log_level, err);
    }
};

// ===================================================================
// Engine public API
// ===================================================================
Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

int Engine::initialize(const Tables& tables,
                       int memory_mode,
                       const std::string& engine_cache_path_utf8,
                       int device_id,
                       int log_level,
                       std::string& err) {
    Impl& s = *impl_;
    err.clear();
    s.width = tables.width;
    s.height = tables.height;
    s.pixels = tables.pixels;
    s.pad = tables.width + 4;   // 覆盖 N..N+W+3 的哑元索引
    s.memory_mode = (memory_mode == 1) ? 1 : 0;
    s.device_id = device_id;
    s.log_level = log_level;
    s.table_bytes = tables.bytes();
    s.logger.level = log_level;

    int code = CudaCheck(cudaSetDevice(device_id), "cudaSetDevice", err, UNDIST_ERR_INIT_FAILED);
    if (code != UNDIST_OK) return code;

    cudaDeviceProp prop{};
    code = CudaCheck(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties",
                     err, UNDIST_ERR_INIT_FAILED);
    if (code != UNDIST_OK) return code;
    std::snprintf(s.gpu_name, sizeof(s.gpu_name), "%s", prop.name);
    s.sm_count = prop.multiProcessorCount;

    const int trt_ver = ::getInferLibVersion();
    std::snprintf(s.trt_version, sizeof(s.trt_version), "%d.%d.%d",
                  trt_ver / 10000, (trt_ver / 100) % 100, trt_ver % 100);

    code = CudaCheck(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking),
                     "cudaStreamCreate", err, UNDIST_ERR_INIT_FAILED);
    if (code != UNDIST_OK) return code;

    if (!engine_cache_path_utf8.empty()) {
        s.cache_path = fs::u8path(engine_cache_path_utf8);
    }

    const auto t_engine = Clock::now();
    TrtPtr<nvinfer1::ICudaEngine> eng = s.load_or_build(log_level, err);
    if (!eng) {
        if (err.empty()) err = "TRT 引擎构建失败";
        return UNDIST_ERR_ENGINE_BUILD;
    }
    s.engine = std::move(eng);
    LogMsg(log_level, 1, "engine ready in %.1f ms", ElapsedMs(t_engine, Clock::now()));

    s.context.reset(s.engine->createExecutionContext());
    if (!s.context) {
        err = "createExecutionContext 返回空";
        return UNDIST_ERR_ENGINE_BUILD;
    }

    // IO 契约自检
    const char* expected[] = {"img", "idx0", "fu", "fv", "out_u8"};
    for (const char* nm : expected) {
        const nvinfer1::Dims d = s.engine->getTensorShape(nm);
        if (d.nbDims < 0) {
            err = std::string("引擎缺少张量: ") + nm;
            return UNDIST_ERR_ENGINE_BUILD;
        }
    }

    const auto t_alloc = Clock::now();
    const size_t n = static_cast<size_t>(s.pixels);
    code = CudaCheck(cudaMalloc(reinterpret_cast<void**>(&s.d_idx), n * 4),
                     "cudaMalloc(idx0)", err, UNDIST_ERR_INIT_FAILED);
    if (code == UNDIST_OK)
        code = CudaCheck(cudaMalloc(reinterpret_cast<void**>(&s.d_fu), n * 4),
                         "cudaMalloc(fu)", err, UNDIST_ERR_INIT_FAILED);
    if (code == UNDIST_OK)
        code = CudaCheck(cudaMalloc(reinterpret_cast<void**>(&s.d_fv), n * 4),
                         "cudaMalloc(fv)", err, UNDIST_ERR_INIT_FAILED);
    if (code == UNDIST_OK)
        code = CudaCheck(cudaMalloc(reinterpret_cast<void**>(&s.d_in), n),
                         "cudaMalloc(in)", err, UNDIST_ERR_INIT_FAILED);
    if (code == UNDIST_OK)
        code = CudaCheck(cudaMalloc(reinterpret_cast<void**>(&s.d_out), n),
                         "cudaMalloc(out)", err, UNDIST_ERR_INIT_FAILED);
    if (code != UNDIST_OK) return code;

    const auto t_h2d = Clock::now();
    code = CudaCheck(cudaMemcpy(s.d_idx, tables.idx0.data(), n * 4, cudaMemcpyHostToDevice),
                     "cudaMemcpy(idx0)", err, UNDIST_ERR_INIT_FAILED);
    if (code == UNDIST_OK)
        code = CudaCheck(cudaMemcpy(s.d_fu, tables.fu.data(), n * 4, cudaMemcpyHostToDevice),
                         "cudaMemcpy(fu)", err, UNDIST_ERR_INIT_FAILED);
    if (code == UNDIST_OK)
        code = CudaCheck(cudaMemcpy(s.d_fv, tables.fv.data(), n * 4, cudaMemcpyHostToDevice),
                         "cudaMemcpy(fv)", err, UNDIST_ERR_INIT_FAILED);
    if (code != UNDIST_OK) return code;
    const double table_ms = ElapsedMs(t_h2d, Clock::now());
    LogMsg(log_level, 1, "tables uploaded: %.1f MB in %.1f ms (%.1f GB/s)",
           double(s.table_bytes) / (1024.0 * 1024.0), table_ms,
           table_ms > 0.0 ? double(s.table_bytes) / (1024.0 * 1024.0 * 1024.0) / (table_ms / 1000.0) : 0.0);

    code = s.allocate_host_staging(err);
    if (code != UNDIST_OK) return code;

    if (!s.context->setTensorAddress("img", s.d_in) ||
        !s.context->setTensorAddress("idx0", s.d_idx) ||
        !s.context->setTensorAddress("fu", s.d_fu) ||
        !s.context->setTensorAddress("fv", s.d_fv) ||
        !s.context->setTensorAddress("out_u8", s.d_out)) {
        err = "setTensorAddress 失败（张量名与引擎不匹配）";
        return UNDIST_ERR_ENGINE_BUILD;
    }

    LogMsg(log_level, 1, "init done in %.1f ms (device=%.1f MB, mode=%s)",
           ElapsedMs(t_engine, Clock::now()),
           double(s.table_bytes + int64_t(2) * s.pixels) / (1024.0 * 1024.0),
           s.memory_mode == 1 ? "pinned" : "pageable");
    (void)t_alloc;
    return UNDIST_OK;
}

int Engine::infer(const uint8_t* src, int src_stride,
                  uint8_t* dst, int dst_stride,
                  bool host_pinned,
                  undist_timing& timing, std::string& err) {
    Impl& s = *impl_;
    if (!s.context || !s.stream) {
        err = "引擎未就绪";
        return UNDIST_ERR_NOT_READY;
    }
    const size_t n = static_cast<size_t>(s.pixels);
    const auto t0 = Clock::now();
    auto h2d = [&](const uint8_t* host, size_t host_pitch) -> cudaError_t {
        if (host_pitch == static_cast<size_t>(s.width)) {
            return cudaMemcpyAsync(s.d_in, host, n, cudaMemcpyHostToDevice, s.stream);
        }
        return cudaMemcpy2DAsync(s.d_in, s.width, host, host_pitch, s.width, s.height,
                                 cudaMemcpyHostToDevice, s.stream);
    };
    auto d2h = [&](uint8_t* host, size_t host_pitch) -> cudaError_t {
        if (host_pitch == static_cast<size_t>(s.width)) {
            return cudaMemcpyAsync(host, s.d_out, n, cudaMemcpyDeviceToHost, s.stream);
        }
        return cudaMemcpy2DAsync(host, host_pitch, s.d_out, s.width, s.width, s.height,
                                 cudaMemcpyDeviceToHost, s.stream);
    };

    // 路径 A：调用方缓冲已是页锁内存 —— 零拷贝、直接异步搬运（工程推荐）
    if (host_pinned) {
        const auto t1 = Clock::now();
        int code = CudaCheck(h2d(src, static_cast<size_t>(src_stride)), "H2D(async)",
                             err, UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        code = CudaCheck(cudaStreamSynchronize(s.stream), "H2D sync", err,
                         UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        const auto t2 = Clock::now();

        if (!s.context->enqueueV3(s.stream)) {
            err = "enqueueV3 失败";
            return UNDIST_ERR_INFERENCE_FAILED;
        }
        code = CudaCheck(cudaStreamSynchronize(s.stream), "infer sync", err,
                         UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        const auto t3 = Clock::now();

        code = CudaCheck(d2h(dst, static_cast<size_t>(dst_stride)), "D2H(async)",
                         err, UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        code = CudaCheck(cudaStreamSynchronize(s.stream), "D2H sync", err,
                         UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        const auto t4 = Clock::now();

        timing.h2d_ms = ElapsedMs(t1, t2);
        timing.infer_ms = ElapsedMs(t2, t3);
        timing.d2h_ms = ElapsedMs(t3, t4);
        timing.total_ms = ElapsedMs(t0, t4);
        return UNDIST_OK;
    }

    // 路径 B：pageable 源 + 库内 pinned 暂存
    if (s.memory_mode == 1 && s.h_in && s.h_out) {
        CopyRows(src, src_stride, s.h_in, s.width, s.width, s.height);

        const auto t1 = Clock::now();
        int code = CudaCheck(h2d(s.h_in, static_cast<size_t>(s.width)), "H2D memcpy",
                             err, UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        code = CudaCheck(cudaStreamSynchronize(s.stream), "H2D sync", err,
                         UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        const auto t2 = Clock::now();

        if (!s.context->enqueueV3(s.stream)) {
            err = "enqueueV3 失败";
            return UNDIST_ERR_INFERENCE_FAILED;
        }
        code = CudaCheck(cudaStreamSynchronize(s.stream), "infer sync", err,
                         UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        const auto t3 = Clock::now();

        code = CudaCheck(d2h(s.h_out, static_cast<size_t>(s.width)), "D2H memcpy",
                         err, UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        code = CudaCheck(cudaStreamSynchronize(s.stream), "D2H sync", err,
                         UNDIST_ERR_INFERENCE_FAILED);
        if (code != UNDIST_OK) return code;
        const auto t4 = Clock::now();

        CopyRows(s.h_out, s.width, dst, dst_stride, s.width, s.height);
        const auto t5 = Clock::now();

        timing.h2d_ms = ElapsedMs(t1, t2);
        timing.infer_ms = ElapsedMs(t2, t3);
        timing.d2h_ms = ElapsedMs(t3, t4);
        timing.total_ms = ElapsedMs(t0, t5);
        return UNDIST_OK;
    }

    // 路径 C：pageable 直传（驱动内部中转；连续行时走 cudaMemcpy 快路径）
    const auto t1 = Clock::now();
    int code = UNDIST_OK;
    if (src_stride == s.width) {
        code = CudaCheck(cudaMemcpy(s.d_in, src, n, cudaMemcpyHostToDevice),
                         "H2D memcpy", err, UNDIST_ERR_INFERENCE_FAILED);
    } else {
        code = CudaCheck(cudaMemcpy2D(s.d_in, s.width, src, static_cast<size_t>(src_stride),
                                      s.width, s.height, cudaMemcpyHostToDevice),
                         "H2D memcpy2D", err, UNDIST_ERR_INFERENCE_FAILED);
    }
    if (code != UNDIST_OK) return code;
    const auto t2 = Clock::now();

    if (!s.context->enqueueV3(s.stream)) {
        err = "enqueueV3 失败";
        return UNDIST_ERR_INFERENCE_FAILED;
    }
    code = CudaCheck(cudaStreamSynchronize(s.stream), "infer sync", err, UNDIST_ERR_INFERENCE_FAILED);
    if (code != UNDIST_OK) return code;
    const auto t3 = Clock::now();

    if (dst_stride == s.width) {
        code = CudaCheck(cudaMemcpy(dst, s.d_out, n, cudaMemcpyDeviceToHost),
                         "D2H memcpy", err, UNDIST_ERR_INFERENCE_FAILED);
    } else {
        code = CudaCheck(cudaMemcpy2D(dst, static_cast<size_t>(dst_stride), s.d_out, s.width,
                                      s.width, s.height, cudaMemcpyDeviceToHost),
                         "D2H memcpy2D", err, UNDIST_ERR_INFERENCE_FAILED);
    }
    if (code != UNDIST_OK) return code;
    const auto t4 = Clock::now();

    timing.h2d_ms = ElapsedMs(t1, t2);
    timing.infer_ms = ElapsedMs(t2, t3);
    timing.d2h_ms = ElapsedMs(t3, t4);
    timing.total_ms = ElapsedMs(t1, t4);
    return UNDIST_OK;
}

int Engine::bench_kernel(const uint8_t* src, int iterations,
                         undist_timing& timing, std::string& err) {
    Impl& s = *impl_;
    if (!s.context || !s.stream) {
        err = "引擎未就绪";
        return UNDIST_ERR_NOT_READY;
    }
    if (!src || iterations < 1) {
        err = "bench_kernel 参数非法";
        return UNDIST_ERR_INVALID_ARG;
    }
    const size_t n = static_cast<size_t>(s.pixels);
    int code = CudaCheck(cudaMemcpy2D(s.d_in, s.width, src, s.width, s.width, s.height,
                                      cudaMemcpyHostToDevice),
                         "H2D memcpy2D", err, UNDIST_ERR_INFERENCE_FAILED);
    if (code != UNDIST_OK) return code;

    // warmup
    if (!s.context->enqueueV3(s.stream)) {
        err = "enqueueV3 失败(warmup)";
        return UNDIST_ERR_INFERENCE_FAILED;
    }
    code = CudaCheck(cudaStreamSynchronize(s.stream), "warmup sync", err, UNDIST_ERR_INFERENCE_FAILED);
    if (code != UNDIST_OK) return code;

    const auto t0 = Clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (!s.context->enqueueV3(s.stream)) {
            err = "enqueueV3 失败";
            return UNDIST_ERR_INFERENCE_FAILED;
        }
    }
    code = CudaCheck(cudaStreamSynchronize(s.stream), "bench sync", err, UNDIST_ERR_INFERENCE_FAILED);
    if (code != UNDIST_OK) return code;
    const double total = ElapsedMs(t0, Clock::now());

    timing.h2d_ms = 0.0;
    timing.d2h_ms = 0.0;
    timing.infer_ms = total / iterations;
    timing.total_ms = timing.infer_ms;
    return UNDIST_OK;
}

int Engine::set_memory_mode(int memory_mode, std::string& err) {
    Impl& s = *impl_;
    const int mode = (memory_mode == 1) ? 1 : 0;
    if (mode == s.memory_mode) return UNDIST_OK;
    if (mode == 0) {
        if (s.h_in) { cudaFreeHost(s.h_in); s.h_in = nullptr; }
        if (s.h_out) { cudaFreeHost(s.h_out); s.h_out = nullptr; }
        s.memory_mode = 0;
        LogMsg(s.log_level, 1, "memory mode -> pageable");
        return UNDIST_OK;
    }
    s.memory_mode = 1;
    const int code = s.allocate_host_staging(err);
    if (code != UNDIST_OK) return code;
    LogMsg(s.log_level, 1, "memory mode -> pinned");
    return UNDIST_OK;
}

int Engine::memory_mode() const { return impl_->memory_mode; }
int Engine::width() const { return impl_->width; }
int Engine::height() const { return impl_->height; }
int64_t Engine::table_bytes() const { return impl_->table_bytes; }
int64_t Engine::engine_bytes() const { return impl_->engine_bytes; }
int64_t Engine::device_bytes() const {
    return impl_->table_bytes + int64_t(2) * impl_->pixels;
}
const char* Engine::gpu_name() const { return impl_->gpu_name; }
int Engine::sm_count() const { return impl_->sm_count; }
const char* Engine::trt_version() const { return impl_->trt_version; }

} // namespace undist

