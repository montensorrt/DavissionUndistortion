// ===================================================================
// test_undist_api.cpp —— 去畸变 V1（TRT）验收 / 性能 / 内存模式对照
//
// 口径与 CPU 版本完全对等：内存进 → 内存出（不含文件 I/O 计时）。
//
// 验收标准（对 Algorithms/undistortion_standaloneV0/demo/output0 的 CPU Release 产物）：
//   1) 最大绝对差 <= 1 LSB
//   2) 差异像素比例 <= 1e-5
//   3) 差 >= 2 的像素数 == 0
//   4) PSNR >= 60 dB
//
// 输出：
//   - demo/output-trt/*.bmp        去畸变结果
//   - document/data/*.json         验收/性能数据
// ===================================================================
#include "undist_api.h"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

// ---------------- 参数 ----------------
struct Args {
    std::string matrix;
    std::string raw;
    std::string cpu;
    std::string out;
    std::string json;
    std::string engine_cache;
    int limit = 0;              // 0 = 全部
    int iters = 3;              // 每图计时次数
    int pageable_images = 8;    // pageable 对照使用的图片数
    int kernel_iters = 60;
    int log_level = 1;
};

void Usage() {
    std::printf(
        "test_undist_api --matrix <dir> --raw <dir> --cpu <dir> --out <dir> --json <file>\n"
        "                [--engine-cache <file>] [--limit N] [--iters K]\n"
        "                [--pageable-images M] [--kernel-iters L] [--log 0|1|2]\n");
}

bool ParseArgs(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) return false;
            dst = argv[++i];
            return true;
        };
        auto next_int = [&](int& dst) {
            if (i + 1 >= argc) return false;
            dst = std::atoi(argv[++i]);
            return true;
        };
        if (k == "--matrix") { if (!next(a.matrix)) return false; }
        else if (k == "--raw") { if (!next(a.raw)) return false; }
        else if (k == "--cpu") { if (!next(a.cpu)) return false; }
        else if (k == "--out") { if (!next(a.out)) return false; }
        else if (k == "--json") { if (!next(a.json)) return false; }
        else if (k == "--engine-cache") { if (!next(a.engine_cache)) return false; }
        else if (k == "--limit") { if (!next_int(a.limit)) return false; }
        else if (k == "--iters") { if (!next_int(a.iters)) return false; }
        else if (k == "--pageable-images") { if (!next_int(a.pageable_images)) return false; }
        else if (k == "--kernel-iters") { if (!next_int(a.kernel_iters)) return false; }
        else if (k == "--log") { if (!next_int(a.log_level)) return false; }
        else if (k == "--help" || k == "-h") { Usage(); return false; }
        else { std::fprintf(stderr, "未知参数: %s\n", k.c_str()); return false; }
    }
    return !a.matrix.empty() && !a.raw.empty() && !a.cpu.empty() && !a.out.empty() && !a.json.empty();
}

// ---------------- 统计 ----------------
struct Stats {
    int n = 0;
    double mean = 0, min = 0, max = 0, p50 = 0, p90 = 0;
};

Stats ComputeStats(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n = static_cast<int>(v.size());
    s.min = v.front();
    s.max = v.back();
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    s.p50 = v[v.size() / 2];
    s.p90 = v[std::min(v.size() - 1, static_cast<size_t>(v.size() * 0.9))];
    return s;
}

struct ImageCompare {
    int64_t diff_pixels = 0;
    int max_abs_diff = 0;
    int64_t count_ge2 = 0;
    double mae = 0.0;
    double psnr_db = 0.0;
};

ImageCompare CompareGray(const cv::Mat& a, const cv::Mat& b) {
    ImageCompare r;
    CV_Assert(a.type() == CV_8U && b.type() == CV_8U && a.size() == b.size());
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    r.diff_pixels = cv::countNonZero(diff);
    double maxv = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxv);
    r.max_abs_diff = static_cast<int>(maxv);
    r.count_ge2 = cv::countNonZero(diff > 1);
    const double pixels = static_cast<double>(a.total());
    r.mae = cv::sum(diff)[0] / pixels;
    cv::Mat d32;
    diff.convertTo(d32, CV_32F);
    const double mse = cv::sum(d32.mul(d32))[0] / pixels;
    r.psnr_db = mse <= 0.0 ? 999.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
    return r;
}

// ---------------- JSON 辅助 ----------------
std::string JStr(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else {
                    o += static_cast<char>(c);
                }
        }
    }
    o += "\"";
    return o;
}

std::string JNum(double v, int prec = 6) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*f", prec, v);
    return buf;
}

std::string JStats(const Stats& s) {
    std::ostringstream o;
    o << "{\"n\":" << s.n
      << ",\"mean\":" << JNum(s.mean)
      << ",\"min\":" << JNum(s.min)
      << ",\"max\":" << JNum(s.max)
      << ",\"p50\":" << JNum(s.p50)
      << ",\"p90\":" << JNum(s.p90) << "}";
    return o.str();
}

struct ImageRow {
    std::string name;
    int width = 0, height = 0;
    ImageCompare cmp;
    bool deterministic = true;
    std::vector<double> h2d, infer, d2h, total;
};

struct ContractCase {
    std::string name;
    int expected = 0;
    int actual = 0;
    bool passed = false;
    std::string note;
};

// ---------------- 图像工具 ----------------
bool LoadGray(const fs::path& p, cv::Mat& out) {
    out = cv::imread(p.string(), cv::IMREAD_GRAYSCALE);
    return !out.empty();
}

void LogLine(int level, int need, const char* fmt, ...) {
    if (level < need) return;
    va_list ap;
    va_start(ap, fmt);
    std::vprintf(fmt, ap);
    std::printf("\n");
    va_end(ap);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, args)) {
        Usage();
        return 2;
    }

    std::error_code ec;
    fs::create_directories(args.out, ec);
    fs::create_directories(fs::path(args.json).parent_path(), ec);

    // 输入清单：以 raw 目录为准
    std::vector<std::string> names;
    for (const auto& e : fs::directory_iterator(args.raw, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string ext = e.path().extension().string();
        if (ext == ".bmp" || ext == ".BMP" || ext == ".png" || ext == ".PNG") {
            names.push_back(e.path().filename().string());
        }
    }
    std::sort(names.begin(), names.end());
    if (args.limit > 0 && static_cast<int>(names.size()) > args.limit) {
        names.resize(args.limit);
    }
    if (names.empty()) {
        std::fprintf(stderr, "raw 目录没有图像: %s\n", args.raw.c_str());
        return 3;
    }

    cv::Mat first;
    if (!LoadGray(fs::u8path(args.raw) / names.front(), first)) {
        std::fprintf(stderr, "首图读取失败: %s\n", names.front().c_str());
        return 3;
    }
    const int W = first.cols;
    const int H = first.rows;
    std::printf("[accept] %zu images, %dx%d, iters=%d\n", names.size(), W, H, args.iters);

    // ---------------- 初始化（pinned） ----------------
    undist_options opt{};
    opt.memory_mode = 1;
    opt.device_id = 0;
    opt.channels = 1;
    opt.log_level = args.log_level;

    undist_ctx* ctx = nullptr;
    const auto t_init0 = Clock::now();
    int code = undist_create(args.matrix.c_str(), W, H, &opt,
                             args.engine_cache.empty() ? nullptr : args.engine_cache.c_str(),
                             &ctx);
    const double init_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_init0).count();
    if (code != UNDIST_OK) {
        std::fprintf(stderr, "undist_create 失败: %d / %s\n", code, undist_get_error(ctx));
        if (ctx) undist_release(ctx);
        return 4;
    }
    undist_info info{};
    undist_get_info(ctx, &info);
    std::printf("[init] %.1f ms (tables %.1f + engine %.1f), table=%.1f MB, device=%.1f MB, gpu=%s, trt=%s\n",
                init_ms, info.init_table_ms, info.init_engine_ms,
                info.table_bytes / 1048576.0, info.device_bytes / 1048576.0,
                info.gpu_name, info.tensorrt_version);

    // ---------------- 逐图验收 ----------------
    std::vector<ImageRow> rows;
    rows.reserve(names.size());
    for (size_t idx = 0; idx < names.size(); ++idx) {
        const std::string& name = names[idx];
        cv::Mat raw, cpu;
        if (!LoadGray(fs::u8path(args.raw) / name, raw)) {
            std::fprintf(stderr, "读取失败(raw): %s\n", name.c_str());
            return 5;
        }
        if (!LoadGray(fs::u8path(args.cpu) / name, cpu)) {
            std::fprintf(stderr, "读取失败(cpu): %s\n", name.c_str());
            return 5;
        }
        if (raw.cols != W || raw.rows != H || cpu.cols != W || cpu.rows != H) {
            std::fprintf(stderr, "尺寸不一致: %s\n", name.c_str());
            return 5;
        }

        cv::Mat out(H, W, CV_8U);
        // warmup
        code = undist_process(ctx, raw.data, static_cast<int>(raw.step), out.data,
                              static_cast<int>(out.step), 0, nullptr);
        if (code != UNDIST_OK) {
            std::fprintf(stderr, "process 失败: %s / %d / %s\n", name.c_str(), code,
                         undist_get_error(ctx));
            return 6;
        }
        cv::Mat first_out = out.clone();

        ImageRow row;
        row.name = name;
        row.width = W;
        row.height = H;
        for (int k = 0; k < args.iters; ++k) {
            undist_timing t{};
            code = undist_process(ctx, raw.data, static_cast<int>(raw.step), out.data,
                                  static_cast<int>(out.step), 0, &t);
            if (code != UNDIST_OK) {
                std::fprintf(stderr, "process 失败: %s / %d\n", name.c_str(), code);
                return 6;
            }
            row.h2d.push_back(t.h2d_ms);
            row.infer.push_back(t.infer_ms);
            row.d2h.push_back(t.d2h_ms);
            row.total.push_back(t.total_ms);
        }
        if (cv::norm(out, first_out, cv::NORM_INF) != 0.0) row.deterministic = false;
        row.cmp = CompareGray(out, cpu);
        rows.push_back(std::move(row));

        if (!cv::imwrite((fs::u8path(args.out) / name).string(), out)) {
            std::fprintf(stderr, "写出失败: %s\n", name.c_str());
            return 7;
        }

        const ImageRow& r = rows.back();
        std::printf("[accept] %-14s diff=%8lld  max=%d  ge2=%lld  mae=%.3e  psnr=%7.2f dB\n",
                    r.name.c_str(), static_cast<long long>(r.cmp.diff_pixels),
                    r.cmp.max_abs_diff, static_cast<long long>(r.cmp.count_ge2),
                    r.cmp.mae, r.cmp.psnr_db);
    }

    // ---------------- 汇总 ----------------
    int64_t total_diff = 0, total_pixels = 0, total_ge2 = 0;
    int max_abs = 0;
    bool all_deterministic = true;
    double psnr_min = 999.0;
    std::vector<double> all_h2d, all_infer, all_d2h, all_total;
    for (const auto& r : rows) {
        total_diff += r.cmp.diff_pixels;
        total_pixels += static_cast<int64_t>(r.width) * r.height;
        total_ge2 += r.cmp.count_ge2;
        max_abs = std::max(max_abs, r.cmp.max_abs_diff);
        psnr_min = std::min(psnr_min, r.cmp.psnr_db);
        all_deterministic = all_deterministic && r.deterministic;
        all_h2d.insert(all_h2d.end(), r.h2d.begin(), r.h2d.end());
        all_infer.insert(all_infer.end(), r.infer.begin(), r.infer.end());
        all_d2h.insert(all_d2h.end(), r.d2h.begin(), r.d2h.end());
        all_total.insert(all_total.end(), r.total.begin(), r.total.end());
    }
    const double diff_ratio = total_pixels > 0 ? double(total_diff) / double(total_pixels) : 0.0;

    const Stats s_h2d = ComputeStats(all_h2d);
    const Stats s_infer = ComputeStats(all_infer);
    const Stats s_d2h = ComputeStats(all_d2h);
    const Stats s_total = ComputeStats(all_total);

    const bool acc_max = (max_abs <= 1);
    const bool acc_ratio = (diff_ratio <= 1e-5);
    const bool acc_ge2 = (total_ge2 == 0);
    const bool acc_psnr = (psnr_min >= 60.0);
    bool acceptance_passed = acc_max && acc_ratio && acc_ge2 && acc_psnr && all_deterministic;

    // ---------------- kernel-only ----------------
    cv::Mat kraw;
    LoadGray(fs::u8path(args.raw) / names.front(), kraw);
    undist_timing kernel_t{};
    code = undist_bench_kernel(ctx, kraw.data, args.kernel_iters, &kernel_t);
    if (code != UNDIST_OK) {
        std::fprintf(stderr, "bench_kernel 失败: %d / %s\n", code, undist_get_error(ctx));
        return 8;
    }

    // ---------------- pageable 对照（同 ctx 切模式，不重传表） ----------------
    std::vector<double> pg_h2d, pg_infer, pg_d2h, pg_total;
    const int pg_n = std::min<int>(args.pageable_images, static_cast<int>(rows.size()));
    code = undist_set_memory_mode(ctx, 0);
    if (code == UNDIST_OK) {
        for (int i = 0; i < pg_n; ++i) {
            cv::Mat raw;
            LoadGray(fs::u8path(args.raw) / rows[i].name, raw);
            cv::Mat out(H, W, CV_8U);
            for (int k = 0; k < args.iters; ++k) {
                undist_timing t{};
                code = undist_process(ctx, raw.data, static_cast<int>(raw.step), out.data,
                                      static_cast<int>(out.step), 0, &t);
                if (code != UNDIST_OK) break;
                pg_h2d.push_back(t.h2d_ms);
                pg_infer.push_back(t.infer_ms);
                pg_d2h.push_back(t.d2h_ms);
                pg_total.push_back(t.total_ms);
            }
        }
        undist_set_memory_mode(ctx, 1);
    }
    const Stats p_h2d = ComputeStats(pg_h2d);
    const Stats p_infer = ComputeStats(pg_infer);
    const Stats p_d2h = ComputeStats(pg_d2h);
    const Stats p_total = ComputeStats(pg_total);

    // ---------------- 零拷贝 pinned（调用方页锁缓冲；工程推荐路径） ----------------
    std::vector<double> zc_h2d, zc_infer, zc_d2h, zc_total;
    bool zero_copy_ok = false;
    uint8_t* pin_in = nullptr;
    uint8_t* pin_out = nullptr;
    cudaError_t e_in = cudaHostAlloc(reinterpret_cast<void**>(&pin_in), size_t(W) * H,
                                     cudaHostAllocDefault);
    cudaError_t e_out = cudaHostAlloc(reinterpret_cast<void**>(&pin_out), size_t(W) * H,
                                      cudaHostAllocDefault);
    if (e_in == cudaSuccess && e_out == cudaSuccess) {
        zero_copy_ok = true;
        for (int i = 0; i < pg_n && zero_copy_ok; ++i) {
            cv::Mat raw, cpu;
            LoadGray(fs::u8path(args.raw) / rows[i].name, raw);
            LoadGray(fs::u8path(args.cpu) / rows[i].name, cpu);
            std::memcpy(pin_in, raw.data, size_t(W) * H);
            for (int k = 0; k < args.iters; ++k) {
                undist_timing t{};
                code = undist_process(ctx, pin_in, W, pin_out, W, 1, &t);
                if (code != UNDIST_OK) { zero_copy_ok = false; break; }
                zc_h2d.push_back(t.h2d_ms);
                zc_infer.push_back(t.infer_ms);
                zc_d2h.push_back(t.d2h_ms);
                zc_total.push_back(t.total_ms);
            }
            if (!zero_copy_ok) break;
            // 结果正确性：与 CPU 产物比对，应与 staging 路径完全一致
            cv::Mat pinned_view(H, W, CV_8U, pin_out);
            const ImageCompare c = CompareGray(pinned_view, cpu);
            if (c.diff_pixels != rows[i].cmp.diff_pixels || c.max_abs_diff != rows[i].cmp.max_abs_diff) {
                std::fprintf(stderr, "零拷贝 pinned 结果与 staging 路径不一致: %s\n",
                             rows[i].name.c_str());
                zero_copy_ok = false;
            }
        }
    }
    if (pin_in) cudaFreeHost(pin_in);
    if (pin_out) cudaFreeHost(pin_out);
    const Stats z_h2d = ComputeStats(zc_h2d);
    const Stats z_infer = ComputeStats(zc_infer);
    const Stats z_d2h = ComputeStats(zc_d2h);
    const Stats z_total = ComputeStats(zc_total);

    // ---------------- C ABI 契约用例 ----------------
    std::vector<ContractCase> contracts;
    auto run_case = [&](const std::string& name, int expected,
                        const std::function<int()>& fn) {
        ContractCase c;
        c.name = name;
        c.expected = expected;
        c.actual = fn();
        c.passed = (c.actual == expected);
        contracts.push_back(c);
    };

    run_case("create_null_dir", UNDIST_ERR_INVALID_ARG, [&] {
        undist_ctx* c = nullptr;
        return undist_create(nullptr, W, H, nullptr, nullptr, &c);
    });
    run_case("create_missing_dir", UNDIST_ERR_MATRIX_MISSING, [&] {
        undist_ctx* c = nullptr;
        const int rc = undist_create("Z:/__not_exist__/undist", W, H, nullptr, nullptr, &c);
        if (c) {
            if (std::strlen(undist_get_error(c)) == 0) return -999;
            undist_release(c);
        }
        return rc;
    });
    run_case("create_wrong_size", UNDIST_ERR_MATRIX_SIZE, [&] {
        undist_ctx* c = nullptr;
        const int rc = undist_create(args.matrix.c_str(), 64, 64, nullptr, nullptr, &c);
        if (c) undist_release(c);
        return rc;
    });
    run_case("create_channels3", UNDIST_ERR_UNSUPPORTED, [&] {
        undist_ctx* c = nullptr;
        undist_options o{};
        o.channels = 3;
        o.memory_mode = 1;
        const int rc = undist_create(args.matrix.c_str(), W, H, &o, nullptr, &c);
        if (c) undist_release(c);
        return rc;
    });
    run_case("process_null_dst", UNDIST_ERR_INVALID_IMAGE, [&] {
        cv::Mat raw;
        LoadGray(fs::u8path(args.raw) / names.front(), raw);
        return undist_process(ctx, raw.data, static_cast<int>(raw.step), nullptr, W, 0, nullptr);
    });
    run_case("process_bad_stride", UNDIST_ERR_INVALID_IMAGE, [&] {
        cv::Mat raw, out(H, W, CV_8U);
        LoadGray(fs::u8path(args.raw) / names.front(), raw);
        return undist_process(ctx, raw.data, W / 2, out.data, static_cast<int>(out.step), 0, nullptr);
    });
    run_case("set_memory_mode_invalid", UNDIST_ERR_INVALID_ARG, [&] {
        return undist_set_memory_mode(ctx, 7);
    });
    run_case("get_info_null", UNDIST_ERR_INVALID_ARG, [&] {
        return undist_get_info(ctx, nullptr);
    });
    run_case("release_null", UNDIST_OK, [&] {
        undist_release(nullptr);
        return 0;
    });

    bool contracts_passed = true;
    for (const auto& c : contracts) contracts_passed = contracts_passed && c.passed;
    acceptance_passed = acceptance_passed && contracts_passed;

    // ---------------- JSON ----------------
    std::ofstream jf(args.json, std::ios::binary | std::ios::trunc);
    if (!jf) {
        std::fprintf(stderr, "JSON 写出失败: %s\n", args.json.c_str());
        return 9;
    }
    std::ostringstream j;
    j << "{\n";
    j << "  \"meta\": {\n";
    j << "    \"suite\": \"undistortionV1/trt\",\n";
    j << "    \"timestamp\": " << JStr([] {
        char buf[32];
        std::time_t now = std::time(nullptr);
        std::tm tmv{};
        localtime_s(&tmv, &now);
        std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
        return std::string(buf);
    }()) << ",\n";
    j << "    \"images\": " << rows.size() << ",\n";
    j << "    \"width\": " << W << ", \"height\": " << H << ",\n";
    j << "    \"iters\": " << args.iters << ",\n";
    j << "    \"kernel_iters\": " << args.kernel_iters << ",\n";
    j << "    \"matrix_dir\": " << JStr(args.matrix) << ",\n";
    j << "    \"raw_dir\": " << JStr(args.raw) << ",\n";
    j << "    \"cpu_dir\": " << JStr(args.cpu) << ",\n";
    j << "    \"out_dir\": " << JStr(args.out) << ",\n";
    j << "    \"engine_cache\": " << JStr(args.engine_cache) << "\n";
    j << "  },\n";
    j << "  \"env\": {\n";
    j << "    \"gpu\": " << JStr(info.gpu_name) << ",\n";
    j << "    \"sm_count\": " << info.sm_count << ",\n";
    j << "    \"tensorrt\": " << JStr(info.tensorrt_version) << ",\n";
    j << "    \"table_variant\": " << info.table_variant << ",\n";
    j << "    \"table_bytes_per_pixel\": " << JNum(info.table_bytes_per_pixel, 1) << "\n";
    j << "  },\n";
    j << "  \"init\": {\n";
    j << "    \"table_ms\": " << JNum(info.init_table_ms) << ",\n";
    j << "    \"engine_ms\": " << JNum(info.init_engine_ms) << ",\n";
    j << "    \"total_ms\": " << JNum(info.init_total_ms) << ",\n";
    j << "    \"table_bytes\": " << info.table_bytes << ",\n";
    j << "    \"engine_bytes\": " << info.engine_bytes << ",\n";
    j << "    \"device_bytes\": " << info.device_bytes << ",\n";
    j << "    \"pinned_host_bytes\": " << (2 * int64_t(W) * H) << "\n";
    j << "  },\n";
    j << "  \"acceptance\": {\n";
    j << "    \"passed\": " << (acceptance_passed ? "true" : "false") << ",\n";
    j << "    \"criteria\": {\"max_abs_diff\":1,\"diff_ratio\":1e-05,\"count_ge2\":0,\"psnr_db\":60.0},\n";
    j << "    \"checks\": {\"max_abs\": " << (acc_max ? "true" : "false")
      << ", \"diff_ratio\": " << (acc_ratio ? "true" : "false")
      << ", \"count_ge2\": " << (acc_ge2 ? "true" : "false")
      << ", \"psnr\": " << (acc_psnr ? "true" : "false")
      << ", \"deterministic\": " << (all_deterministic ? "true" : "false") << "},\n";
    j << "    \"totals\": {\"diff_pixels\": " << total_diff
      << ", \"pixels\": " << total_pixels
      << ", \"diff_ratio\": " << JNum(diff_ratio, 9)
      << ", \"max_abs_diff\": " << max_abs
      << ", \"count_ge2\": " << total_ge2
      << ", \"psnr_min_db\": " << JNum(psnr_min, 3) << "},\n";
    j << "    \"images_identical\": " << std::count_if(rows.begin(), rows.end(),
        [](const ImageRow& r) { return r.cmp.diff_pixels == 0; }) << "\n";
    j << "  },\n";
    j << "  \"timing_pinned\": {\n";
    j << "    \"h2d_ms\": " << JStats(s_h2d) << ",\n";
    j << "    \"infer_ms\": " << JStats(s_infer) << ",\n";
    j << "    \"d2h_ms\": " << JStats(s_d2h) << ",\n";
    j << "    \"total_ms\": " << JStats(s_total) << "\n";
    j << "  },\n";
    j << "  \"timing_pageable\": {\n";
    j << "    \"images\": " << pg_n << ",\n";
    j << "    \"h2d_ms\": " << JStats(p_h2d) << ",\n";
    j << "    \"infer_ms\": " << JStats(p_infer) << ",\n";
    j << "    \"d2h_ms\": " << JStats(p_d2h) << ",\n";
    j << "    \"total_ms\": " << JStats(p_total) << "\n";
    j << "  },\n";
    j << "  \"timing_pinned_zero_copy\": {\n";
    j << "    \"available\": " << (zero_copy_ok ? "true" : "false") << ",\n";
    j << "    \"images\": " << pg_n << ",\n";
    j << "    \"h2d_ms\": " << JStats(z_h2d) << ",\n";
    j << "    \"infer_ms\": " << JStats(z_infer) << ",\n";
    j << "    \"d2h_ms\": " << JStats(z_d2h) << ",\n";
    j << "    \"total_ms\": " << JStats(z_total) << "\n";
    j << "  },\n";
    j << "  \"kernel_only\": {\"iters\": " << args.kernel_iters
      << ", \"infer_ms\": " << JNum(kernel_t.infer_ms)
      << ", \"table_gbs\": " << JNum(kernel_t.infer_ms > 0
            ? (double(info.table_bytes) / 1e9) / (kernel_t.infer_ms / 1000.0) : 0.0, 3) << "},\n";
    j << "  \"memory_modes\": {\n";
    j << "    \"pinned_total_ms\": " << JNum(s_total.mean) << ",\n";
    j << "    \"pageable_total_ms\": " << JNum(p_total.mean) << ",\n";
    j << "    \"zero_copy_total_ms\": " << JNum(z_total.mean) << ",\n";
    j << "    \"speedup_pinned_vs_pageable\": " << JNum(p_total.mean > 0 ? p_total.mean / s_total.mean : 0.0, 3) << ",\n";
    j << "    \"speedup_zero_copy_vs_pageable\": " << JNum(z_total.mean > 0 ? p_total.mean / z_total.mean : 0.0, 3) << ",\n";
    j << "    \"staging_host_copy_ms\": " << JNum(s_total.mean > 0 ? s_total.mean - z_total.mean : 0.0, 3) << ",\n";
    j << "    \"h2d_ratio\": " << JNum(p_h2d.mean > 0 ? p_h2d.mean / std::max(1e-9, s_h2d.mean) : 0.0, 3) << ",\n";
    j << "    \"d2h_ratio\": " << JNum(p_d2h.mean > 0 ? p_d2h.mean / std::max(1e-9, s_d2h.mean) : 0.0, 3) << ",\n";
    j << "    \"kernel_only_ms\": " << JNum(kernel_t.infer_ms) << ",\n";
    j << "    \"pcie_pinned_share\": " << JNum(s_total.mean > 0 ? (s_h2d.mean + s_d2h.mean) / s_total.mean : 0.0, 4) << ",\n";
    j << "    \"pcie_pageable_share\": " << JNum(p_total.mean > 0 ? (p_h2d.mean + p_d2h.mean) / p_total.mean : 0.0, 4) << ",\n";
    j << "    \"kernel_share_zero_copy\": " << JNum(z_total.mean > 0 ? z_infer.mean / z_total.mean : 0.0, 4) << "\n";
    j << "  },\n";
    j << "  \"images\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const ImageRow& r = rows[i];
        const Stats h = ComputeStats(r.h2d), in = ComputeStats(r.infer),
                    d = ComputeStats(r.d2h), t = ComputeStats(r.total);
        j << "    {\"name\":" << JStr(r.name)
          << ",\"diff_pixels\":" << r.cmp.diff_pixels
          << ",\"max_abs_diff\":" << r.cmp.max_abs_diff
          << ",\"count_ge2\":" << r.cmp.count_ge2
          << ",\"mae\":" << JNum(r.cmp.mae, 9)
          << ",\"psnr_db\":" << JNum(r.cmp.psnr_db, 3)
          << ",\"deterministic\":" << (r.deterministic ? "true" : "false")
          << ",\"h2d_ms\":" << JNum(h.mean)
          << ",\"infer_ms\":" << JNum(in.mean)
          << ",\"d2h_ms\":" << JNum(d.mean)
          << ",\"total_ms\":" << JNum(t.mean) << "}";
        if (i + 1 != rows.size()) j << ",";
        j << "\n";
    }
    j << "  ],\n";
    j << "  \"contract_tests\": [\n";
    for (size_t i = 0; i < contracts.size(); ++i) {
        const auto& c = contracts[i];
        j << "    {\"name\":" << JStr(c.name) << ",\"expected\":" << c.expected
          << ",\"actual\":" << c.actual << ",\"passed\":" << (c.passed ? "true" : "false") << "}";
        if (i + 1 != contracts.size()) j << ",";
        j << "\n";
    }
    j << "  ]\n";
    j << "}\n";
    jf << j.str();
    jf.close();

    // 逐图 CSV（便于报表）
    const fs::path csv = fs::path(args.json).parent_path() / "acceptance_per_image.csv";
    std::ofstream cf(csv, std::ios::binary | std::ios::trunc);
    cf << "name,diff_pixels,max_abs_diff,count_ge2,mae,psnr_db,h2d_ms,infer_ms,d2h_ms,total_ms\n";
    for (const auto& r : rows) {
        const Stats h = ComputeStats(r.h2d), in = ComputeStats(r.infer),
                    d = ComputeStats(r.d2h), t = ComputeStats(r.total);
        cf << r.name << "," << r.cmp.diff_pixels << "," << r.cmp.max_abs_diff << ","
           << r.cmp.count_ge2 << "," << r.cmp.mae << "," << r.cmp.psnr_db << ","
           << h.mean << "," << in.mean << "," << d.mean << "," << t.mean << "\n";
    }
    cf.close();

    std::printf("\n[result] %s\n", acceptance_passed ? "PASS" : "FAIL");
    std::printf("  diff=%lld/%lld (%.3e), max=%d, ge2=%lld, psnr_min=%.2f dB, identical=%lld/%zu\n",
                static_cast<long long>(total_diff), static_cast<long long>(total_pixels),
                diff_ratio, max_abs, static_cast<long long>(total_ge2), psnr_min,
                static_cast<long long>(std::count_if(rows.begin(), rows.end(),
                    [](const ImageRow& r) { return r.cmp.diff_pixels == 0; })), rows.size());
    std::printf("  pinned+stage: h2d=%.3f infer=%.3f d2h=%.3f total=%.3f ms (%.1f img/s)\n",
                s_h2d.mean, s_infer.mean, s_d2h.mean, s_total.mean,
                s_total.mean > 0 ? 1000.0 / s_total.mean : 0.0);
    std::printf("  pageable: h2d=%.3f infer=%.3f d2h=%.3f total=%.3f ms (%.1f img/s)\n",
                p_h2d.mean, p_infer.mean, p_d2h.mean, p_total.mean,
                p_total.mean > 0 ? 1000.0 / p_total.mean : 0.0);
    std::printf("  pinned-0copy: h2d=%.3f infer=%.3f d2h=%.3f total=%.3f ms (%.1f img/s)\n",
                z_h2d.mean, z_infer.mean, z_d2h.mean, z_total.mean,
                z_total.mean > 0 ? 1000.0 / z_total.mean : 0.0);
    std::printf("  kernel  : %.3f ms (%.1f GB/s table bandwidth)\n", kernel_t.infer_ms,
                kernel_t.infer_ms > 0 ? (double(info.table_bytes) / 1e9) / (kernel_t.infer_ms / 1000.0) : 0.0);
    std::printf("  init    : %.1f ms (tables %.1f + engine %.1f)\n", init_ms,
                info.init_table_ms, info.init_engine_ms);
    std::printf("  json    : %s\n", args.json.c_str());

    undist_release(ctx);
    return acceptance_passed ? 0 : 1;
}
