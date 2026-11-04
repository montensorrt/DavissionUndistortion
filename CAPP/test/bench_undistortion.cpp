// ===================================================================
// bench_undistortion.cpp
// -------------------------------------------------------------------
// Undistortion Standalone V0 —— Release 模式基准测试与一致性校验程序
//
// 设计目标（不修改库源码的前提下，获得可审计的数据）：
//   1) production 模式：调用正式接口 ImageUndistorter::infer 处理 demo/raw，
//      结果写入 demo/output0，并统计各阶段墙钟耗时（Release）。
//   2) replica  模式：在测试程序内按库的语义逐行复刻算法流水线，
//      - 与 production 产物逐字节比对，证明"复刻实现 == 正式实现"；
//      - 给出算法内部各阶段拆分耗时（U8->F32 / min-max 扫描 / 4-tap gather /
//        量化 / 输出 min-max 扫描）；
//      - 给出"FMA 链 vs mul+add 链"的逐字节差异，用于评估 TRT 侧是否
//        必须严格复刻 FMA 语义；
//      - 可选导出量化前的 float32 结果，供 GPU(TRT) 侧做逐位置比对。
//
// 说明：本文件刻意把阶段计时放在"复刻实现"而非污染库源码，
//       并用 bit-exact 比对来担保计时的代表性与正确性。
// ===================================================================

#include "Undistortion.h"
#include "Logger.h"

#include <opencv2/opencv.hpp>
#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static inline double ToMs(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// ===================================================================
// 统计工具
// ===================================================================
struct Stats {
    size_t n = 0;
    double mean = 0.0, minv = 0.0, maxv = 0.0, median = 0.0;
    double p90 = 0.0, p99 = 0.0, stddev = 0.0, trimmed = 0.0;
};

static double Percentile(const std::vector<double>& sortedAsc, double p) {
    if (sortedAsc.empty()) return 0.0;
    if (sortedAsc.size() == 1) return sortedAsc[0];
    double pos = (p / 100.0) * static_cast<double>(sortedAsc.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(pos));
    size_t hi = static_cast<size_t>(std::ceil(pos));
    double t = pos - static_cast<double>(lo);
    return sortedAsc[lo] * (1.0 - t) + sortedAsc[hi] * t;
}

static Stats ComputeStats(const std::vector<double>& v) {
    Stats s;
    s.n = v.size();
    if (v.empty()) return s;

    std::vector<double> sortedAsc = v;
    std::sort(sortedAsc.begin(), sortedAsc.end());

    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    s.minv = sortedAsc.front();
    s.maxv = sortedAsc.back();
    s.median = Percentile(sortedAsc, 50.0);
    s.p90 = Percentile(sortedAsc, 90.0);
    s.p99 = Percentile(sortedAsc, 99.0);

    double acc = 0.0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.stddev = (v.size() > 1) ? std::sqrt(acc / static_cast<double>(v.size() - 1)) : 0.0;

    // 与原始 test_undistortion.cpp 相同的口径：去掉一个最小值和一个最大值
    if (sortedAsc.size() < 3) {
        s.trimmed = s.mean;
    } else {
        double sum = 0.0;
        for (size_t i = 1; i + 1 < sortedAsc.size(); ++i) sum += sortedAsc[i];
        s.trimmed = sum / static_cast<double>(sortedAsc.size() - 2);
    }
    return s;
}

static std::string Fmt(double v, int prec = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << v;
    return oss.str();
}

// ===================================================================
// 命令行参数
// ===================================================================
struct Options {
    std::string mode = "both";     // production | replica | both
    std::string input = "";
    std::string matrix = "";
    std::string output = "";
    std::string jsonPath = "";
    std::string csvPath = "";
    int reps = 5;
    int warmup = 2;
    int limit = 0;
    bool save = true;
    int dumpFloat = 0;             // 前 N 张图导出量化前 float32（逐位置比对用）
    std::string dumpDir = "";
    std::string logLevel = "warn"; // warn | info | debug
};

static Options g_opt;

// 多线程扩展性测量的线程档位
static const int kMtThreads[] = { 1, 4, 8, 16, 20 };
static const size_t kMtCount = sizeof(kMtThreads) / sizeof(kMtThreads[0]);

static bool ParseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        auto nextInt = [&](int& dst) {
            if (i + 1 >= argc) return false;
            dst = std::atoi(argv[++i]);
            return true;
        };
        if (a == "--mode") { if (!next(g_opt.mode)) return false; }
        else if (a == "--input") { if (!next(g_opt.input)) return false; }
        else if (a == "--matrix") { if (!next(g_opt.matrix)) return false; }
        else if (a == "--output") { if (!next(g_opt.output)) return false; }
        else if (a == "--json") { if (!next(g_opt.jsonPath)) return false; }
        else if (a == "--csv") { if (!next(g_opt.csvPath)) return false; }
        else if (a == "--reps") { if (!nextInt(g_opt.reps)) return false; }
        else if (a == "--warmup") { if (!nextInt(g_opt.warmup)) return false; }
        else if (a == "--limit") { if (!nextInt(g_opt.limit)) return false; }
        else if (a == "--save") { int v = 1; if (!nextInt(v)) return false; g_opt.save = (v != 0); }
        else if (a == "--dump-float") { if (!nextInt(g_opt.dumpFloat)) return false; }
        else if (a == "--dump-dir") { if (!next(g_opt.dumpDir)) return false; }
        else if (a == "--log-level") { if (!next(g_opt.logLevel)) return false; }
        else if (a == "--help" || a == "-h") { return false; }
        else {
            std::cerr << "Unknown argument: " << a << std::endl;
            return false;
        }
    }
    return true;
}

static void PrintUsage() {
    std::cout <<
        "bench_undistortion --mode production|replica|both [options]\n"
        "  --input  DIR   raw image dir      (default: <proj>/demo/raw)\n"
        "  --matrix DIR   12 x .bin dir      (default: <proj>/botong25)\n"
        "  --output DIR   result dir         (default: <proj>/demo/output0)\n"
        "  --json   PATH  metrics json out\n"
        "  --csv    PATH  per-image csv out\n"
        "  --reps   N     measured iterations per image (default 5)\n"
        "  --warmup N     warmup iterations           (default 2)\n"
        "  --limit  N     process only first N images (0 = all)\n"
        "  --save   0|1   write result images         (default 1)\n"
        "  --dump-float N dump pre-quantization f32 of first N images\n"
        "  --dump-dir DIR where to put the .f32 dumps\n"
        "  --log-level warn|info|debug               (default warn)\n";
}

// ===================================================================
// 图像枚举
// ===================================================================
static bool IsImageExt(const std::string& ext) {
    static const char* kExts[] = { ".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff" };
    for (const char* e : kExts) if (ext == e) return true;
    return false;
}

static std::vector<std::string> CollectImages(const std::string& root) {
    std::vector<std::string> out;
    for (const auto& e : fs::recursive_directory_iterator(root)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (IsImageExt(ext)) out.push_back(e.path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// ===================================================================
// 日志
// ===================================================================
static void InitLogger(const std::string& level) {
    LogConfig cfg;
    cfg.enableConsole = (level == "debug");
    cfg.enableFile = false;          // 避免磁盘 IO 干扰计时
    if (level == "debug") cfg.level = LogLevel::DEBUG;
    else if (level == "info") cfg.level = LogLevel::INFO;
    else cfg.level = LogLevel::WARN;
    cfg.logPath[0] = '\0';
    Logger::getInstance().initialize(cfg);
}

// ===================================================================
// 复刻实现（replica）
// -------------------------------------------------------------------
// 严格按 src/Undistortion.cpp 的语义实现，用于：
//   - 与正式实现逐字节比对
//   - 阶段耗时拆分
//   - 导出量化前 float32 供 GPU 逐位置比对
// ===================================================================
class ReplicaUndistorter {
public:
    bool Init(const std::string& matrixPath, int W, int H, double& loadMs, double& buildMs) {
        W_ = W; H_ = H;
        const int N = W * H;

        auto t0 = Clock::now();
        std::vector<std::vector<float>> U(4), V(4), A(4);
        const char* uNames[4] = { "U_1.bin", "U_2.bin", "U_3.bin", "U_4.bin" };
        const char* vNames[4] = { "V_1.bin", "V_2.bin", "V_3.bin", "V_4.bin" };
        const char* aNames[4] = { "A1.bin", "A2.bin", "A3.bin", "A4.bin" };
        for (int k = 0; k < 4; ++k) {
            if (!LoadSimpleMatrix(W, H, matrixPath + "/" + uNames[k], U[k])) return false;
            if (!LoadSimpleMatrix(W, H, matrixPath + "/" + vNames[k], V[k])) return false;
            if (!LoadSimpleMatrix(W, H, matrixPath + "/" + aNames[k], A[k])) return false;
        }
        auto t1 = Clock::now();
        loadMs = ToMs(t1 - t0);

        // 预计算索引与权重（与 precomputeIndices 完全一致）
        idx0_.resize(N); idx1_.resize(N); idx2_.resize(N); idx3_.resize(N);
        w0_.resize(N); w1_.resize(N); w2_.resize(N); w3_.resize(N);

        const int dummyIndex = N;
        for (int v = 0; v < H; ++v) {
            for (int u = 0; u < W; ++u) {
                const int idx = v * W + u;
                const int u0 = static_cast<int>(U[0][idx]), v0 = static_cast<int>(V[0][idx]);
                const int u1 = static_cast<int>(U[1][idx]), v1 = static_cast<int>(V[1][idx]);
                const int u2 = static_cast<int>(U[2][idx]), v2 = static_cast<int>(V[2][idx]);
                const int u3 = static_cast<int>(U[3][idx]), v3 = static_cast<int>(V[3][idx]);

                const bool ok0 = (u0 >= 1 && v0 >= 1 && u0 < W && v0 < H);
                const bool ok1 = (u1 >= 1 && v1 >= 1 && u1 < W && v1 < H);
                const bool ok2 = (u2 >= 1 && v2 >= 1 && u2 < W && v2 < H);
                const bool ok3 = (u3 >= 1 && v3 >= 1 && u3 < W && v3 < H);
                const bool allValid = ok0 && ok1 && ok2 && ok3;

                if (allValid) {
                    idx0_[idx] = v0 * W + u0;
                    idx1_[idx] = v1 * W + u1;
                    idx2_[idx] = v2 * W + u2;
                    idx3_[idx] = v3 * W + u3;
                } else {
                    idx0_[idx] = idx1_[idx] = idx2_[idx] = idx3_[idx] = dummyIndex;
                    ++invalidCount_;
                }
                w0_[idx] = A[0][idx]; w1_[idx] = A[1][idx];
                w2_[idx] = A[2][idx]; w3_[idx] = A[3][idx];
            }
        }
        auto t2 = Clock::now();
        buildMs = ToMs(t2 - t1);

        U.clear(); V.clear(); A.clear();
        inF_.assign(static_cast<size_t>(N) + 1, 0.0f);
        tmp_.assign(static_cast<size_t>(N), 0.0f);
        out_.assign(static_cast<size_t>(N), 0u);
        return true;
    }

    int Width() const { return W_; }
    int Height() const { return H_; }
    int PixelCount() const { return W_ * H_; }
    size_t InvalidCount() const { return invalidCount_; }

    void ConvertIn(const uint8_t* src) {
        const int N = W_ * H_;
        std::copy(src, src + N, inF_.begin());
        inF_[N] = 0.0f;   // 哑像素
    }

    struct ScanResult { float minv; float maxv; };

    ScanResult ScanInput() const {
        const int N = W_ * H_;
        float mn = *std::min_element(inF_.begin(), inF_.begin() + N);
        float mx = *std::max_element(inF_.begin(), inF_.begin() + N);
        return { mn, mx };
    }

    ScanResult ScanOutput() const {
        const int N = W_ * H_;
        uint8_t mn = *std::min_element(out_.begin(), out_.begin() + N);
        uint8_t mx = *std::max_element(out_.begin(), out_.begin() + N);
        return { static_cast<float>(mn), static_cast<float>(mx) };
    }

    // 4-tap gather + FMA 链（与 simdUndistortAVX2_FMA 一致）
    void GatherFma() { GatherRange(0, W_ * H_); }

    void GatherRange(int beg, int end) {
        const int N = W_ * H_;
        const float* in = inF_.data();
        float* dst = tmp_.data();
        const int* p0 = idx0_.data(); const int* p1 = idx1_.data();
        const int* p2 = idx2_.data(); const int* p3 = idx3_.data();
        const float* q0 = w0_.data(); const float* q1 = w1_.data();
        const float* q2 = w2_.data(); const float* q3 = w3_.data();

        int i = beg;
        for (; i <= end - 8; i += 8) {
            __m256i i0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + i));
            __m256i i1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + i));
            __m256i i2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + i));
            __m256i i3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p3 + i));
            __m256 s0 = _mm256_i32gather_ps(in, i0, 4);
            __m256 s1 = _mm256_i32gather_ps(in, i1, 4);
            __m256 s2 = _mm256_i32gather_ps(in, i2, 4);
            __m256 s3 = _mm256_i32gather_ps(in, i3, 4);
            __m256 v0 = _mm256_loadu_ps(q0 + i);
            __m256 v1 = _mm256_loadu_ps(q1 + i);
            __m256 v2 = _mm256_loadu_ps(q2 + i);
            __m256 v3 = _mm256_loadu_ps(q3 + i);
            __m256 r = _mm256_fmadd_ps(s0, v0,
                        _mm256_fmadd_ps(s1, v1,
                        _mm256_fmadd_ps(s2, v2,
                        _mm256_mul_ps(s3, v3))));
            _mm256_storeu_ps(dst + i, r);
        }
        for (; i < end; ++i) {
            int a0 = (i < static_cast<int>(idx0_.size())) ? idx0_[i] : 0;
            int a1 = (i < static_cast<int>(idx1_.size())) ? idx1_[i] : 0;
            int a2 = (i < static_cast<int>(idx2_.size())) ? idx2_[i] : 0;
            int a3 = (i < static_cast<int>(idx3_.size())) ? idx3_[i] : 0;
            float x0 = (a0 >= 0 && a0 < N) ? in[a0] : 0.0f;
            float x1 = (a1 >= 0 && a1 < N) ? in[a1] : 0.0f;
            float x2 = (a2 >= 0 && a2 < N) ? in[a2] : 0.0f;
            float x3 = (a3 >= 0 && a3 < N) ? in[a3] : 0.0f;
            dst[i] = x0 * q0[i] + x1 * q1[i] + x2 * q2[i] + x3 * q3[i];
        }
    }

    // 与 GatherFma 数学等价，但显式使用 mul+add（评估 FMA 语义敏感性）
    void GatherMulAdd() {
        const int N = W_ * H_;
        const float* in = inF_.data();
        float* dst = tmp_.data();
        const int* p0 = idx0_.data(); const int* p1 = idx1_.data();
        const int* p2 = idx2_.data(); const int* p3 = idx3_.data();
        const float* q0 = w0_.data(); const float* q1 = w1_.data();
        const float* q2 = w2_.data(); const float* q3 = w3_.data();

        int i = 0;
        for (; i <= N - 8; i += 8) {
            __m256i i0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p0 + i));
            __m256i i1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p1 + i));
            __m256i i2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p2 + i));
            __m256i i3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p3 + i));
            __m256 s0 = _mm256_i32gather_ps(in, i0, 4);
            __m256 s1 = _mm256_i32gather_ps(in, i1, 4);
            __m256 s2 = _mm256_i32gather_ps(in, i2, 4);
            __m256 s3 = _mm256_i32gather_ps(in, i3, 4);
            __m256 v0 = _mm256_loadu_ps(q0 + i);
            __m256 v1 = _mm256_loadu_ps(q1 + i);
            __m256 v2 = _mm256_loadu_ps(q2 + i);
            __m256 v3 = _mm256_loadu_ps(q3 + i);
            __m256 acc = _mm256_mul_ps(s0, v0);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(s1, v1));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(s2, v2));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(s3, v3));
            _mm256_storeu_ps(dst + i, acc);
        }
        for (; i < N; ++i) {
            dst[i] = in[idx0_[i]] * q0[i] + in[idx1_[i]] * q1[i]
                   + in[idx2_[i]] * q2[i] + in[idx3_[i]] * q3[i];
        }
    }

    // clamp(0,255) + 0.5 + trunc（与 convertFloatToByteAVX2 一致）
    void Quantize() { QuantizeRange(0, W_ * H_); }

    void QuantizeRange(int beg, int end) {
        const float* src = tmp_.data();
        uint8_t* dst = out_.data();
        const __m256 vZero = _mm256_setzero_ps();
        const __m256 vMax = _mm256_set1_ps(255.0f);
        const __m256 vHalf = _mm256_set1_ps(0.5f);
        int i = beg;
        for (; i <= end - 8; i += 8) {
            __m256 vf = _mm256_loadu_ps(src + i);
            vf = _mm256_max_ps(vf, vZero);
            vf = _mm256_min_ps(vf, vMax);
            vf = _mm256_add_ps(vf, vHalf);
            __m256i vi = _mm256_cvttps_epi32(vf);
            int lanes[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), vi);
            for (int k = 0; k < 8; ++k) dst[i + k] = static_cast<uint8_t>(lanes[k]);
        }
        for (; i < end; ++i) {
            float v = src[i];
            if (v < 0.0f) v = 0.0f; else if (v > 255.0f) v = 255.0f;
            dst[i] = static_cast<uint8_t>(v + 0.5f);
        }
    }

    // 与库内 infer() 一一对应的完整流水线（用于与 production 的 infer 对齐）
    void FullInfer(const uint8_t* src) {
        ConvertIn(src);
        volatile float dummyIn = ScanInput().maxv; (void)dummyIn;
        GatherFma();
        Quantize();
        volatile float dummyOut = ScanOutput().maxv; (void)dummyOut;
    }

    // 多线程版：仅做 gather + quantize（算法核心），按 8 像素对齐切分区间
    void GatherQuantMT(int threads) {
        const int N = W_ * H_;
        const int chunk = ((N / threads) + 7) / 8 * 8;
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            const int beg = t * chunk;
            if (beg >= N) break;
            const int end = std::min(N, beg + chunk);
            pool.emplace_back([this, beg, end]() { GatherRange(beg, end); });
        }
        for (auto& th : pool) th.join();
        QuantizeRange(0, N);
    }

    const std::vector<uint8_t>& Output() const { return out_; }

    // 量化前的浮点结果（用于与 GPU 的 pre-clip 结果做逐位置比对）
    const std::vector<float>& TmpFloat() const { return tmp_; }

private:
    // 与 loadSimpleMatrix 一致：文件为列主序，转成行主序
    static bool LoadSimpleMatrix(int W, int H, const std::string& path, std::vector<float>& out) {
        const size_t count = static_cast<size_t>(W) * H;
        std::vector<float> colMajor(count);
        std::ifstream fin(path, std::ios::binary);
        if (!fin) {
            std::cerr << "Cannot open matrix file: " << path << std::endl;
            return false;
        }
        fin.read(reinterpret_cast<char*>(colMajor.data()), count * sizeof(float));
        if (fin.gcount() != static_cast<std::streamsize>(count * sizeof(float))) {
            std::cerr << "Matrix file size mismatch: " << path << std::endl;
            return false;
        }
        out.resize(count);
        for (int col = 0; col < W; ++col) {
            const int colBase = col * H;
            for (int row = 0; row < H; ++row) {
                out[static_cast<size_t>(row) * W + col] = colMajor[colBase + row];
            }
        }
        return true;
    }

    int W_ = 0, H_ = 0;
    size_t invalidCount_ = 0;
    std::vector<int> idx0_, idx1_, idx2_, idx3_;
    std::vector<float> w0_, w1_, w2_, w3_;
    std::vector<float> inF_, tmp_;
    std::vector<uint8_t> out_;
};

// ===================================================================
// 逐图记录
// ===================================================================
struct ImageRecord {
    std::string name;
    double readMs = 0, copyMs = 0, inferMs = 0, inferLibMs = 0, writeMs = 0;
    double cvtMs = 0, scanInMs = 0, gatherMs = 0, quantMs = 0, scanOutMs = 0;
    double fullMs = 0;
    std::vector<double> mtMs;   // 多线程 gather+quant 的逐档中位耗时
    double fmaDiffPixels = 0, fmaMaxDiff = 0;
    long long diffPixels = -1;
    int maxAbsDiff = -1;
    bool ok = false;
};

// ===================================================================
// JSON / CSV 输出
// ===================================================================
static void WriteJson(const std::string& path,
                      const std::map<std::string, Stats>& prod,
                      const std::map<std::string, Stats>& repl,
                      const std::string& metaJson) {
    if (path.empty()) return;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());

    std::ofstream f(path, std::ios::binary);
    f << "{\n";
    f << "  \"meta\": " << metaJson << ",\n";
    f << "  \"stats\": {\n";
    bool firstGroup = true;
    auto emitGroup = [&](const char* group, const std::map<std::string, Stats>& m, bool& firstG) {
        if (m.empty()) return;
        if (!firstG) f << ",\n";
        firstG = false;
        f << "    \"" << group << "\": {\n";
        bool firstItem = true;
        for (const auto& kv : m) {
            if (!firstItem) f << ",\n";
            firstItem = false;
            const Stats& s = kv.second;
            f << "      \"" << kv.first << "\": {";
            f << "\"n\": " << s.n
              << ", \"mean\": " << Fmt(s.mean, 4)
              << ", \"min\": " << Fmt(s.minv, 4)
              << ", \"max\": " << Fmt(s.maxv, 4)
              << ", \"median\": " << Fmt(s.median, 4)
              << ", \"p90\": " << Fmt(s.p90, 4)
              << ", \"p99\": " << Fmt(s.p99, 4)
              << ", \"stddev\": " << Fmt(s.stddev, 4)
              << ", \"trimmed\": " << Fmt(s.trimmed, 4) << "}";
        }
        f << "\n    }";
    };
    emitGroup("production", prod, firstGroup);
    emitGroup("replica", repl, firstGroup);
    f << "\n  }\n}\n";
}

static void WriteCsv(const std::string& path, const std::vector<ImageRecord>& recs) {
    if (path.empty()) return;
    fs::path p(path);
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream f(path, std::ios::binary);
    f << "name,read_ms,copy_ms,infer_ms,infer_lib_ms,write_ms,cvt_ms,scan_in_ms,"
         "gather_ms,quant_ms,scan_out_ms,full_pipeline_ms,diff_pixels,max_abs_diff,"
         "fma_diff_pixels,fma_max_diff\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& r : recs) {
        f << r.name << "," << r.readMs << "," << r.copyMs << "," << r.inferMs << ","
          << r.inferLibMs << "," << r.writeMs << "," << r.cvtMs << "," << r.scanInMs << ","
          << r.gatherMs << "," << r.quantMs << "," << r.scanOutMs << "," << r.fullMs << ","
          << r.diffPixels << "," << r.maxAbsDiff << ","
          << static_cast<long long>(r.fmaDiffPixels) << "," << r.fmaMaxDiff << "\n";
    }
}

// ===================================================================
// main
// ===================================================================
int main(int argc, char** argv) {
    const std::string PROJ =
        "E:/algrithm/project/botong/davissionNew/Algorithms/undistortion_standaloneV0";
    g_opt.input  = PROJ + "/demo/raw";
    g_opt.matrix = PROJ + "/botong25";
    g_opt.output = PROJ + "/demo/output0";

    if (!ParseArgs(argc, argv)) { PrintUsage(); return 2; }

    InitLogger(g_opt.logLevel);

    const int MATRIX_W = 5120;
    const int MATRIX_H = 4096;
    const int N = MATRIX_W * MATRIX_H;

    std::vector<std::string> images = CollectImages(g_opt.input);
    if (g_opt.limit > 0 && static_cast<int>(images.size()) > g_opt.limit) {
        images.resize(g_opt.limit);
    }
    if (images.empty()) {
        std::cerr << "No images found in " << g_opt.input << std::endl;
        return 1;
    }

    std::cout << "=== undistortion bench (Release) ===" << std::endl;
    std::cout << "mode       : " << g_opt.mode << std::endl;
    std::cout << "images     : " << images.size() << std::endl;
    std::cout << "input      : " << g_opt.input << std::endl;
    std::cout << "matrix     : " << g_opt.matrix << std::endl;
    std::cout << "output     : " << g_opt.output << std::endl;
    std::cout << "reps/warmup: " << g_opt.reps << "/" << g_opt.warmup << std::endl;
    std::cout << "size       : " << MATRIX_W << "x" << MATRIX_H
              << " (" << Fmt(N / 1.0e6, 2) << " MPix)" << std::endl;

    std::vector<ImageRecord> records(images.size());
    std::map<std::string, Stats> prodStats;
    std::map<std::string, Stats> replStats;

    double setConfigMs = 0.0;
    double replicaLoadMs = 0.0, replicaBuildMs = 0.0;
    size_t replicaInvalid = 0;

    // -------------------------------------------------------------
    // production 模式
    // -------------------------------------------------------------
    if (g_opt.mode == "production" || g_opt.mode == "both") {
        ImageUndistorter undistorter;
        UndistortionConfig cfg;
        std::strncpy(cfg.matrixPath, g_opt.matrix.c_str(), sizeof(cfg.matrixPath) - 1);
        cfg.matrixWidth = MATRIX_W;
        cfg.matrixHeight = MATRIX_H;
        cfg.outputChannels = 1;

        auto t0 = Clock::now();
        ErrorCode rc = undistorter.setConfig(cfg);
        setConfigMs = ToMs(Clock::now() - t0);
        if (rc != ErrorCode::INFER_SUCCESS) {
            std::cerr << "setConfig failed: " << static_cast<int>(rc) << std::endl;
            return 1;
        }
        std::cout << "\n[production] setConfig: " << Fmt(setConfigMs, 1) << " ms" << std::endl;

        if (g_opt.save) fs::create_directories(g_opt.output);

        std::vector<uint8_t> inBuf(N), outBuf(N);
        BaseResult result;

        std::vector<double> vRead, vCopy, vInfer, vInferLib, vWrite;

        for (size_t k = 0; k < images.size(); ++k) {
            ImageRecord& rec = records[k];
            rec.name = fs::path(images[k]).filename().string();

            auto ta = Clock::now();
            cv::Mat img = cv::imread(images[k], cv::IMREAD_GRAYSCALE);
            rec.readMs = ToMs(Clock::now() - ta);
            if (img.empty() || img.cols != MATRIX_W || img.rows != MATRIX_H) {
                std::cerr << "bad image: " << images[k] << std::endl;
                continue;
            }

            auto tb = Clock::now();
            std::memcpy(inBuf.data(), img.data, static_cast<size_t>(N));
            rec.copyMs = ToMs(Clock::now() - tb);

            // warmup
            for (int w = 0; w < g_opt.warmup; ++w) {
                undistorter.infer(inBuf.data(), MATRIX_W, MATRIX_H, outBuf.data(), result);
            }

            std::vector<double> inferSamples, inferLibSamples;
            for (int r = 0; r < g_opt.reps; ++r) {
                auto t1 = Clock::now();
                ErrorCode irc = undistorter.infer(inBuf.data(), MATRIX_W, MATRIX_H,
                                                  outBuf.data(), result);
                auto t2 = Clock::now();
                if (irc != ErrorCode::INFER_SUCCESS) {
                    std::cerr << "infer failed on " << rec.name << std::endl;
                    break;
                }
                inferSamples.push_back(ToMs(t2 - t1));
                inferLibSamples.push_back(static_cast<double>(result.processingTime));
            }
            if (inferSamples.empty()) continue;
            rec.inferMs = ComputeStats(inferSamples).median;
            rec.inferLibMs = ComputeStats(inferLibSamples).median;

            if (g_opt.save) {
                fs::path outPath = fs::path(g_opt.output) / rec.name;
                cv::Mat outMat(MATRIX_H, MATRIX_W, CV_8UC1, outBuf.data());
                auto tc = Clock::now();
                bool ok = cv::imwrite(outPath.u8string(), outMat);
                rec.writeMs = ToMs(Clock::now() - tc);
                if (!ok) std::cerr << "imwrite failed: " << outPath << std::endl;
            }

            rec.ok = true;
            vRead.push_back(rec.readMs);
            vCopy.push_back(rec.copyMs);
            vInfer.push_back(rec.inferMs);
            vInferLib.push_back(rec.inferLibMs);
            if (g_opt.save) vWrite.push_back(rec.writeMs);

            std::cout << "[" << (k + 1) << "/" << images.size() << "] " << rec.name
                      << "  read=" << Fmt(rec.readMs, 2)
                      << " copy=" << Fmt(rec.copyMs, 2)
                      << " infer=" << Fmt(rec.inferMs, 2)
                      << " write=" << Fmt(rec.writeMs, 2) << " ms" << std::endl;
        }

        prodStats["read"] = ComputeStats(vRead);
        prodStats["copy"] = ComputeStats(vCopy);
        prodStats["infer"] = ComputeStats(vInfer);
        prodStats["infer_lib_reported"] = ComputeStats(vInferLib);
        if (!vWrite.empty()) prodStats["write"] = ComputeStats(vWrite);

        std::vector<double> vE2E;
        for (const auto& r : records) {
            if (r.ok) vE2E.push_back(r.readMs + r.copyMs + r.inferMs + r.writeMs);
        }
        prodStats["e2e"] = ComputeStats(vE2E);
    }

    // -------------------------------------------------------------
    // replica 模式
    // -------------------------------------------------------------
    if (g_opt.mode == "replica" || g_opt.mode == "both") {
        ReplicaUndistorter rep;
        if (!rep.Init(g_opt.matrix, MATRIX_W, MATRIX_H, replicaLoadMs, replicaBuildMs)) {
            std::cerr << "replica init failed" << std::endl;
            return 1;
        }
        replicaInvalid = rep.InvalidCount();
        std::cout << "\n[replica] matrix load: " << Fmt(replicaLoadMs, 1)
                  << " ms, index build: " << Fmt(replicaBuildMs, 1) << " ms"
                  << ", out-of-bounds pixels: " << replicaInvalid << std::endl;

        std::vector<double> vCvt, vScanIn, vGather, vQuant, vScanOut, vFull;

        for (size_t k = 0; k < images.size(); ++k) {
            ImageRecord& rec = records[k];
            if (rec.name.empty()) rec.name = fs::path(images[k]).filename().string();

            cv::Mat img = cv::imread(images[k], cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            auto t1 = Clock::now();
            rep.ConvertIn(img.data);
            rec.cvtMs = ToMs(Clock::now() - t1);

            auto t2 = Clock::now();
            rep.ScanInput();
            rec.scanInMs = ToMs(Clock::now() - t2);

            std::vector<double> gs;
            for (int r = 0; r < g_opt.reps; ++r) {
                auto a = Clock::now();
                rep.GatherFma();
                gs.push_back(ToMs(Clock::now() - a));
            }
            rec.gatherMs = ComputeStats(gs).median;

            std::vector<double> qs;
            for (int r = 0; r < g_opt.reps; ++r) {
                auto a = Clock::now();
                rep.Quantize();
                qs.push_back(ToMs(Clock::now() - a));
            }
            rec.quantMs = ComputeStats(qs).median;

            auto t3 = Clock::now();
            rep.ScanOutput();
            rec.scanOutMs = ToMs(Clock::now() - t3);

            // 完整流水线（cvt+scan+4tap+quant+scan）组合计时
            std::vector<double> fulls;
            for (int r = 0; r < g_opt.reps; ++r) {
                auto a = Clock::now();
                rep.FullInfer(img.data);
                fulls.push_back(ToMs(Clock::now() - a));
            }
            rec.fullMs = ComputeStats(fulls).median;

            // 多线程 gather+quant 扩展性（同一张图，按线程数扫描）
            for (size_t ti = 0; ti < kMtCount; ++ti) {
                std::vector<double> ms;
                for (int r = 0; r < 3; ++r) {
                    auto a = Clock::now();
                    rep.GatherQuantMT(kMtThreads[ti]);
                    ms.push_back(ToMs(Clock::now() - a));
                }
                rec.mtMs.push_back(ComputeStats(ms).median);
            }

            // 与 production 产物逐字节比对
            if (g_opt.save) {
                fs::path refPath = fs::path(g_opt.output) / rec.name;
                cv::Mat ref = cv::imread(refPath.u8string(), cv::IMREAD_GRAYSCALE);
                if (!ref.empty() && ref.cols == MATRIX_W && ref.rows == MATRIX_H) {
                    const auto& got = rep.Output();
                    long long diff = 0;
                    int maxd = 0;
                    for (int i = 0; i < N; ++i) {
                        int d = std::abs(static_cast<int>(ref.data[i]) - static_cast<int>(got[i]));
                        if (d) { ++diff; if (d > maxd) maxd = d; }
                    }
                    rec.diffPixels = diff;
                    rec.maxAbsDiff = maxd;
                }
            }

            // FMA 链 vs mul+add 链：仅对首张图统计（评估语义敏感性）
            if (k == 0) {
                std::vector<uint8_t> fmaOut = rep.Output();
                rep.GatherMulAdd();
                rep.Quantize();
                const auto& maOut = rep.Output();
                long long diff = 0;
                int maxd = 0;
                for (int i = 0; i < N; ++i) {
                    int d = std::abs(static_cast<int>(fmaOut[i]) - static_cast<int>(maOut[i]));
                    if (d) { ++diff; if (d > maxd) maxd = d; }
                }
                rec.fmaDiffPixels = static_cast<double>(diff);
                rec.fmaMaxDiff = static_cast<double>(maxd);
                // 还原 FMA 结果
                rep.ConvertIn(img.data);
                rep.GatherFma();
                rep.Quantize();
            }

            vCvt.push_back(rec.cvtMs);
            vScanIn.push_back(rec.scanInMs);
            vGather.push_back(rec.gatherMs);
            vQuant.push_back(rec.quantMs);
            vScanOut.push_back(rec.scanOutMs);
            vFull.push_back(rec.fullMs);

            // 导出量化前 float32 结果（供 GPU 侧做逐位置比对）
            if (static_cast<int>(k) < g_opt.dumpFloat && !g_opt.dumpDir.empty()) {
                fs::create_directories(g_opt.dumpDir);
                fs::path fp = fs::path(g_opt.dumpDir) / (rec.name + ".f32");
                std::ofstream fo(fp, std::ios::binary);
                const auto& tf = rep.TmpFloat();
                fo.write(reinterpret_cast<const char*>(tf.data()),
                         static_cast<std::streamsize>(tf.size() * sizeof(float)));
                std::cout << "      dumped float -> " << fp.string() << std::endl;
            }

            std::cout << "[replica " << (k + 1) << "/" << images.size() << "] " << rec.name
                      << "  cvt=" << Fmt(rec.cvtMs, 2)
                      << " scanIn=" << Fmt(rec.scanInMs, 2)
                      << " gather=" << Fmt(rec.gatherMs, 2)
                      << " quant=" << Fmt(rec.quantMs, 2)
                      << " scanOut=" << Fmt(rec.scanOutMs, 2)
                      << " full=" << Fmt(rec.fullMs, 2)
                      << "  diff=" << rec.diffPixels << " maxdiff=" << rec.maxAbsDiff
                      << std::endl;
        }

        replStats["cvt_in"] = ComputeStats(vCvt);
        replStats["scan_in_minmax"] = ComputeStats(vScanIn);
        replStats["gather_fma"] = ComputeStats(vGather);
        replStats["quantize"] = ComputeStats(vQuant);
        replStats["scan_out_minmax"] = ComputeStats(vScanOut);
        replStats["full_pipeline"] = ComputeStats(vFull);

        std::vector<double> vSum;
        for (const auto& r : records) {
            if (r.gatherMs > 0) {
                vSum.push_back(r.cvtMs + r.scanInMs + r.gatherMs + r.quantMs + r.scanOutMs);
            }
        }
        replStats["replica_infer_sum"] = ComputeStats(vSum);

        // 多线程扩展性
        for (size_t ti = 0; ti < kMtCount; ++ti) {
            std::vector<double> v;
            for (const auto& r : records) {
                if (r.mtMs.size() == kMtCount) v.push_back(r.mtMs[ti]);
            }
            replStats["mt_gather_quant_" + std::to_string(kMtThreads[ti]) + "T"] =
                ComputeStats(v);
        }
    }

    // -------------------------------------------------------------
    // 汇总打印
    // -------------------------------------------------------------
    auto printGroup = [](const char* title, const std::map<std::string, Stats>& m) {
        if (m.empty()) return;
        std::cout << "\n=== " << title << " (ms, n=" << m.begin()->second.n << ") ===" << std::endl;
        std::cout << std::left << std::setw(30) << "stage"
                  << std::right << std::setw(10) << "mean"
                  << std::setw(10) << "median"
                  << std::setw(10) << "min"
                  << std::setw(10) << "max"
                  << std::setw(10) << "p90"
                  << std::setw(10) << "stddev"
                  << std::setw(10) << "trimmed" << std::endl;
        for (const auto& kv : m) {
            const Stats& s = kv.second;
            std::cout << std::left << std::setw(30) << kv.first
                      << std::right << std::setw(10) << Fmt(s.mean, 2)
                      << std::setw(10) << Fmt(s.median, 2)
                      << std::setw(10) << Fmt(s.minv, 2)
                      << std::setw(10) << Fmt(s.maxv, 2)
                      << std::setw(10) << Fmt(s.p90, 2)
                      << std::setw(10) << Fmt(s.stddev, 2)
                      << std::setw(10) << Fmt(s.trimmed, 2) << std::endl;
        }
    };
    printGroup("production (Release)", prodStats);
    printGroup("replica / stage breakdown (Release)", replStats);

    if (!records.empty() && records[0].fmaDiffPixels >= 0) {
        std::cout << "\nFMA vs mul+add (image #0): diff pixels = "
                  << static_cast<long long>(records[0].fmaDiffPixels)
                  << ", max abs diff = " << static_cast<int>(records[0].fmaMaxDiff) << std::endl;
    }

    long long totalDiff = 0;
    int globalMaxDiff = 0;
    bool anyCompared = false;
    for (const auto& r : records) {
        if (r.diffPixels >= 0) {
            anyCompared = true;
            totalDiff += r.diffPixels;
            globalMaxDiff = std::max(globalMaxDiff, r.maxAbsDiff);
        }
    }
    if (anyCompared) {
        std::cout << "bit-exact check (replica vs production, all images): "
                  << "total diff pixels = " << totalDiff
                  << ", max abs diff = " << globalMaxDiff << std::endl;
    }

    // -------------------------------------------------------------
    // 落盘
    // -------------------------------------------------------------
    std::ostringstream meta;
    meta << "{";
    meta << "\"mode\": \"" << g_opt.mode << "\"";
    meta << ", \"images\": " << images.size();
    meta << ", \"width\": " << MATRIX_W << ", \"height\": " << MATRIX_H;
    meta << ", \"pixels\": " << N;
    meta << ", \"reps\": " << g_opt.reps << ", \"warmup\": " << g_opt.warmup;
    meta << ", \"log_level\": \"" << g_opt.logLevel << "\"";
    meta << ", \"set_config_ms\": " << Fmt(setConfigMs, 3);
    meta << ", \"replica_matrix_load_ms\": " << Fmt(replicaLoadMs, 3);
    meta << ", \"replica_index_build_ms\": " << Fmt(replicaBuildMs, 3);
    meta << ", \"out_of_bounds_pixels\": " << replicaInvalid;
    meta << ", \"bit_exact_total_diff_pixels\": " << totalDiff;
    meta << ", \"bit_exact_max_abs_diff\": " << globalMaxDiff;
    meta << "}";

    WriteJson(g_opt.jsonPath, prodStats, replStats, meta.str());
    WriteCsv(g_opt.csvPath, records);

    return 0;
}
