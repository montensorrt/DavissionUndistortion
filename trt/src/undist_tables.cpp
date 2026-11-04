#include "undist_tables.h"
#include "undist_api.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace undist {
namespace {

void LogMsg(int log_level, int need, const char* fmt, ...) {
    if (log_level < need) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[undist][tables] ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}

// 读取列主序 float32 矩阵并转为行主序（与 C++ V0 的 loadSimpleMatrix 完全一致）
int ReadMatrix(const fs::path& path, int width, int height,
               std::vector<float>& out, std::string& err, int log_level) {
    const int64_t count = static_cast<int64_t>(width) * height;
    const int64_t expected_bytes = count * static_cast<int64_t>(sizeof(float));

    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        err = "矩阵文件缺失: " + path.u8string();
        return UNDIST_ERR_MATRIX_MISSING;
    }
    const std::streamoff file_size = fin.tellg();
    if (file_size != expected_bytes) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "矩阵文件大小不符: %s, 期望 %lld 字节 (%d x %d x 4), 实际 %lld 字节",
                      path.u8string().c_str(), static_cast<long long>(expected_bytes),
                      width, height, static_cast<long long>(file_size));
        err = buf;
        return UNDIST_ERR_MATRIX_SIZE;
    }

    fin.seekg(0, std::ios::beg);
    std::vector<float> raw(static_cast<size_t>(count));
    fin.read(reinterpret_cast<char*>(raw.data()), expected_bytes);
    if (!fin) {
        err = "矩阵文件读取失败: " + path.u8string();
        return UNDIST_ERR_MATRIX_MISSING;
    }

    // 分块转置：raw[col * height + row] -> out[row * width + col]
    // 32x32 分块把跨距读写的缓存抖动压到可接受范围（init 一次性开销）
    out.resize(static_cast<size_t>(count));
    constexpr int kBlock = 32;
    for (int c0 = 0; c0 < width; c0 += kBlock) {
        const int c1 = std::min(width, c0 + kBlock);
        for (int r0 = 0; r0 < height; r0 += kBlock) {
            const int r1 = std::min(height, r0 + kBlock);
            for (int r = r0; r < r1; ++r) {
                float* dst_row = out.data() + static_cast<size_t>(r) * width;
                for (int c = c0; c < c1; ++c) {
                    dst_row[c] = raw[static_cast<size_t>(c) * height + r];
                }
            }
        }
    }
    LogMsg(log_level, 2, "loaded %s", path.filename().u8string().c_str());
    return UNDIST_OK;
}

} // namespace

int build_tables(const std::string& matrix_dir_utf8,
                 int width,
                 int height,
                 Tables& out,
                 std::string& err,
                 int log_level) {
    const auto t_begin = std::chrono::steady_clock::now();
    err.clear();

    if (width <= 0 || height <= 0) {
        err = "矩阵尺寸非法";
        return UNDIST_ERR_INVALID_ARG;
    }
    const int64_t pixels = static_cast<int64_t>(width) * height;
    if (pixels <= 0 || pixels > (int64_t(1) << 31) - 16) {
        err = "像素数超出 int32 索引能力";
        return UNDIST_ERR_INVALID_ARG;
    }

    const fs::path dir = fs::u8path(matrix_dir_utf8);
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        err = "矩阵目录不存在: " + matrix_dir_utf8;
        return UNDIST_ERR_MATRIX_MISSING;
    }

    Tables tables;
    tables.width = width;
    tables.height = height;
    tables.pixels = pixels;
    tables.idx0.assign(static_cast<size_t>(pixels), static_cast<int32_t>(pixels));
    tables.fu.assign(static_cast<size_t>(pixels), 0.0f);
    tables.fv.assign(static_cast<size_t>(pixels), 0.0f);

    // ------------------------------------------------------------------
    // 1) 有效位掩码 + idx0（U1/V1）
    //    valid = (u_k in [1,W)) && (v_k in [1,H)) for k = 0..3
    //    逐矩阵累加，峰值内存 ≈ 2 张矩阵 + idx0 + mask，而不是 8 张矩阵
    // ------------------------------------------------------------------
    std::vector<uint8_t> mask(static_cast<size_t>(pixels), 1);
    std::vector<float> u1, v1;
    int rc = ReadMatrix(dir / "U_1.bin", width, height, u1, err, log_level);
    if (rc != UNDIST_OK) return rc;
    rc = ReadMatrix(dir / "V_1.bin", width, height, v1, err, log_level);
    if (rc != UNDIST_OK) return rc;

    {
        int32_t* idx = tables.idx0.data();
        uint8_t* m = mask.data();
        for (int64_t p = 0; p < pixels; ++p) {
            const int u = static_cast<int>(u1[static_cast<size_t>(p)]);
            const int v = static_cast<int>(v1[static_cast<size_t>(p)]);
            if (m[p] && (u < 1 || u >= width || v < 1 || v >= height)) m[p] = 0;
            if (m[p]) idx[p] = static_cast<int32_t>(static_cast<int64_t>(v) * width + u);
        }
    }
    u1.clear();
    u1.shrink_to_fit();
    v1.clear();
    v1.shrink_to_fit();

    for (int k = 2; k <= 4; ++k) {
        std::vector<float> u, v;
        const std::string un = "U_" + std::to_string(k) + ".bin";
        const std::string vn = "V_" + std::to_string(k) + ".bin";
        int tap_rc = ReadMatrix(dir / un, width, height, u, err, log_level);
        if (tap_rc != UNDIST_OK) return tap_rc;
        tap_rc = ReadMatrix(dir / vn, width, height, v, err, log_level);
        if (tap_rc != UNDIST_OK) return tap_rc;
        uint8_t* m = mask.data();
        for (int64_t p = 0; p < pixels; ++p) {
            if (!m[p]) continue;
            const int uk = static_cast<int>(u[static_cast<size_t>(p)]);
            const int vk = static_cast<int>(v[static_cast<size_t>(p)]);
            if (uk < 1 || uk >= width || vk < 1 || vk >= height) m[p] = 0;
        }
        LogMsg(log_level, 2, "tap %d masked", k);
    }

    // 越界像素统一指向哑元 pixelCount（平坦输入尾部补 0）
    int64_t valid = 0;
    for (int64_t p = 0; p < pixels; ++p) {
        if (mask[static_cast<size_t>(p)]) {
            ++valid;
        } else {
            tables.idx0[static_cast<size_t>(p)] = static_cast<int32_t>(pixels);
        }
    }
    tables.valid_pixels = valid;
    tables.out_of_bounds_pixels = pixels - valid;
    mask.clear();
    mask.shrink_to_fit();

    // ------------------------------------------------------------------
    // 2) 权重因子：fu = A2 + A4, fv = A3 + A4
    // ------------------------------------------------------------------
    std::vector<float> a2, a4, a3;
    int rc_a = ReadMatrix(dir / "A2.bin", width, height, a2, err, log_level);
    if (rc_a != UNDIST_OK) return rc_a;
    rc_a = ReadMatrix(dir / "A4.bin", width, height, a4, err, log_level);
    if (rc_a != UNDIST_OK) return rc_a;
    for (int64_t p = 0; p < pixels; ++p) {
        tables.fu[static_cast<size_t>(p)] = a2[static_cast<size_t>(p)] + a4[static_cast<size_t>(p)];
    }
    a2.clear();
    a2.shrink_to_fit();

    rc_a = ReadMatrix(dir / "A3.bin", width, height, a3, err, log_level);
    if (rc_a != UNDIST_OK) return rc_a;
    for (int64_t p = 0; p < pixels; ++p) {
        tables.fv[static_cast<size_t>(p)] = a3[static_cast<size_t>(p)] + a4[static_cast<size_t>(p)];
    }
    a3.clear();
    a3.shrink_to_fit();
    a4.clear();
    a4.shrink_to_fit();

    out = std::move(tables);

    const auto ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_begin).count();
    LogMsg(log_level, 1,
           "tables ready in %.1f ms: %lld px, valid=%lld, out_of_bounds=%lld (%.4f%%), bytes=%.1f MB (12 B/px)",
           ms, static_cast<long long>(pixels), static_cast<long long>(valid),
           static_cast<long long>(tables.out_of_bounds_pixels),
           100.0 * double(tables.out_of_bounds_pixels) / double(pixels),
           double(tables.bytes()) / (1024.0 * 1024.0));
    return UNDIST_OK;
}

} // namespace undist
