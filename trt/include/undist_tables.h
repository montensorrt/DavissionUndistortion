#pragma once

// ===================================================================
// 去畸变紧凑表构建（纯 CPU 层，不依赖 TensorRT / CUDA）
//
// 输入：matrix_dir 下的 12 个 .bin（列主序 float32）
// 输出：变体 F 的 3 张表（共 12 B/px）
//   idx0: int32[N]  第一个抽头的平坦索引；四抽头任一越界时 = N（指向哑元 0）
//   fu  : float[N]  = A2 + A4
//   fv  : float[N]  = A3 + A4
//
// 与 CPU V0 的语义对齐点：
//   1. 四抽头「全有或全无」：任一 tap 越界 => 整个像素读取 0（V0 预计算索引时置哑元）
//   2. 索引取整用 trunc（与 C++ static_cast<int> 一致）
//   3. 权重由双线性因子重建（与 A1..A4 的位级差异 < 1.2e-7，已在原型中审计）
// ===================================================================

#include <cstdint>
#include <string>
#include <vector>

namespace undist {

constexpr int kTableVariantF = 1;

struct Tables {
    int width = 0;
    int height = 0;
    int64_t pixels = 0;

    std::vector<int32_t> idx0;
    std::vector<float> fu;
    std::vector<float> fv;

    int64_t valid_pixels = 0;
    int64_t out_of_bounds_pixels = 0;

    int64_t bytes() const { return pixels * 12; }
};

/**
 * @brief 加载 12 个矩阵并构建变体 F 紧凑表
 * @param matrix_dir UTF-8 目录
 * @param width      矩阵宽度
 * @param height     矩阵高度
 * @param out        输出表（失败时保持未就绪）
 * @param err        失败原因（UTF-8）
 * @param log_level  0/1/2
 * @return undist_error_code
 */
int build_tables(const std::string& matrix_dir,
                 int width,
                 int height,
                 Tables& out,
                 std::string& err,
                 int log_level);

} // namespace undist
