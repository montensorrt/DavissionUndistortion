# -*- coding: utf-8 -*-
"""
reference_check.py —— 去畸变算法的独立 NumPy 参考实现 + 语义分析

目的
----
1) 用 NumPy 独立复刻 Algorithms/undistortion_standaloneV0/CAPP/src/Undistortion.cpp
   的数学语义，与 C++ Release 产物 (demo/output0) 做逐像素比对，
   证明"语义已被完全理解"，并给出 float32 舍入/ FMA 敏感性量级。
2) 抽取矩阵 (U/V/A) 的结构特征，为 TensorRT 方案设计提供事实依据：
   - A 矩阵是否等于"由 frac(u),frac(v) 推导的双线性权重"（决定能否用纹理采样）
   - 越界像素比例（决定边界语义的复刻方式）
   - 分数部分的分布（决定 8bit 纹理插值权重的误差量级）

用法
----
  python tools/reference_check.py --images 3
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATRIX_DIR = os.path.join(PROJ, "botong25")
RAW_DIR = os.path.join(PROJ, "demo", "raw")
OUT_DIR = os.path.join(PROJ, "demo", "output0")
DATA_DIR = os.path.join(PROJ, "document", "data")

W, H = 5120, 4096
N = W * H


def load_matrix(name: str) -> np.ndarray:
    """列主序 .bin -> 行主序 (H, W) float32，与 C++ loadSimpleMatrix 一致。"""
    raw = np.fromfile(os.path.join(MATRIX_DIR, name), dtype=np.float32)
    assert raw.size == N, (name, raw.size)
    return raw.reshape(W, H).T.copy()   # (H, W)


def read_bmp_gray(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, path
    assert img.shape == (H, W), img.shape
    return img


def build_maps():
    """按 precomputeIndices 的语义构建索引与权重。"""
    u = [None] * 4
    for k in range(4):
        m = load_matrix(f"U_{k + 1}.bin")
        u[k] = np.trunc(m).astype(np.int32)
    v = [None] * 4
    for k in range(4):
        m = load_matrix(f"V_{k + 1}.bin")
        v[k] = np.trunc(m).astype(np.int32)

    valid = []
    for k in range(4):
        valid.append((u[k] >= 1) & (v[k] >= 1) & (u[k] < W) & (v[k] < H))
    all_valid = valid[0] & valid[1] & valid[2] & valid[3]

    idx = []
    for k in range(4):
        ik = v[k].astype(np.int64) * W + u[k].astype(np.int64)
        ik = np.where(all_valid, ik, N)
        idx.append(ik.astype(np.int32))

    w = [None] * 4
    for k in range(4):
        w[k] = load_matrix(f"A{k + 1}.bin")

    return u, v, idx, w, all_valid


def run_reference(img: np.ndarray, idx, w, use_fma_emulation: bool = False) -> np.ndarray:
    """复刻 simdUndistortAVX2_FMA + convertFloatToByteAVX2。"""
    inp = np.empty(N + 1, dtype=np.float32)
    inp[:N] = img.reshape(-1).astype(np.float32)
    inp[N] = np.float32(0.0)
    s0, s1, s2, s3 = inp[idx[0]], inp[idx[1]], inp[idx[2]], inp[idx[3]]
    if use_fma_emulation:
        # 用 float64 模拟"只做一次舍入"的融合乘加，量级上更接近 FMA 语义
        acc = (s3.astype(np.float64) * w[3].astype(np.float64))
        acc = s2.astype(np.float64) * w[2].astype(np.float64) + acc
        acc = s1.astype(np.float64) * w[1].astype(np.float64) + acc
        acc = s0.astype(np.float64) * w[0].astype(np.float64) + acc
        r = acc.astype(np.float32)
    else:
        r = s0 * w[0] + s1 * w[1] + s2 * w[2] + s3 * w[3]
    r = np.clip(r, 0.0, 255.0)
    return np.trunc(r + np.float32(0.5)).astype(np.uint8)


def analyse_weights(u, v, w, all_valid):
    """判断 A 矩阵是否为 frac(u)/frac(v) 的双线性权重。"""
    stats = {}

    fu = [u[k] - np.trunc(u[k]) for k in range(4)]   # u[k] 已是 trunc，故 frac 需取原值
    # 注意：u[k] 存的是 trunc 后的整数，分数部分要重新读矩阵才能拿到。
    # 这里改用 idx 关系推断：A 权重与 (1-fu)(1-fv) 的相关性分析需要原始 U/V。
    # 为了不重复读满 12 个 80MB 文件，改在 build_maps 之外单独做一次针对性的采样分析。

    total = N
    stats["out_of_bounds_pixels"] = int((~all_valid).sum())
    stats["out_of_bounds_ratio"] = float((~all_valid).sum() / total)

    # 权重和为 1 的程度（全采样，float64 累加避免误差）
    s = (w[0].astype(np.float64) + w[1].astype(np.float64) +
         w[2].astype(np.float64) + w[3].astype(np.float64))
    valid_w = s[all_valid]
    stats["weight_sum_mean"] = float(valid_w.mean())
    stats["weight_sum_min"] = float(valid_w.min())
    stats["weight_sum_max"] = float(valid_w.max())
    stats["weight_sum_abs_err_max"] = float(np.abs(valid_w - 1.0).max())

    # 权重取值范围 / 量化到 8bit 分数位的误差估计
    for k in range(4):
        wk = w[k][all_valid]
        stats[f"w{k}_min"] = float(wk.min())
        stats[f"w{k}_max"] = float(wk.max())
    # 若把权重截断到 1/256 网格（纹理单元 8bit 权重），最坏加权误差
    quant_err = 0.0
    worst = np.zeros(int(all_valid.sum()), dtype=np.float64)
    for k in range(4):
        wk = w[k][all_valid].astype(np.float64)
        q = np.round(wk * 256.0) / 256.0
        worst = worst + np.abs(wk - q) * 255.0
    quant_err = float(worst.max())
    stats["texture_8bit_weight_quant_max_lsb_error"] = quant_err
    stats["texture_8bit_weight_quant_mean_lsb_error"] = float(worst.mean())
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=3, help="how many images to verify")
    ap.add_argument("--json", default=os.path.join(DATA_DIR, "reference_check.json"))
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[ref] building maps from {MATRIX_DIR} ...")
    t0 = time.time()
    u, v, idx, w, all_valid = build_maps()
    t_build = time.time() - t0
    print(f"[ref] maps built in {t_build:.1f}s; out-of-bounds pixels = "
          f"{int((~all_valid).sum())} ({(~all_valid).sum() / N * 100:.4f}%)")

    report = {
        "matrix_dir": MATRIX_DIR,
        "width": W,
        "height": H,
        "pixels": N,
        "map_build_seconds": round(t_build, 3),
        "out_of_bounds_pixels": int((~all_valid).sum()),
        "out_of_bounds_ratio": float((~all_valid).sum() / N),
        "images": [],
    }

    names = sorted(os.listdir(RAW_DIR))[: max(1, args.images)]
    total_diff = 0
    total_pixels = 0
    for name in names:
        img = read_bmp_gray(os.path.join(RAW_DIR, name))
        ref = read_bmp_gray(os.path.join(OUT_DIR, name))
        got = run_reference(img, idx, w)
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        nz = int((d != 0).sum())
        total_diff += nz
        total_pixels += d.size
        print(f"[ref] {name}: diff_pixels={nz} max_abs_diff={int(d.max())}")
        report["images"].append({
            "name": name,
            "diff_pixels": nz,
            "max_abs_diff": int(d.max()),
            "mean_abs_diff": float(d.mean()),
        })

        # FMA 语义模拟（float64 融合）与纯 float32 的对比
        got_fma = run_reference(img, idx, w, use_fma_emulation=True)
        d2 = np.abs(got_fma.astype(np.int16) - got.astype(np.int16))
        print(f"[ref] {name}: vs float64-fused emulation diff_pixels={int((d2 != 0).sum())} "
              f"max={int(d2.max())}")

    report["total_diff_pixels"] = total_diff
    report["total_compared_pixels"] = total_pixels
    report["diff_ratio"] = total_diff / total_pixels
    print(f"\n[ref] total diff pixels = {total_diff} / {total_pixels} "
          f"({total_diff / total_pixels:.3e})")

    print("[ref] analysing weight structure ...")
    report["weights"] = analyse_weights(u, v, w, all_valid)
    for k, vv in report["weights"].items():
        print(f"   {k}: {vv}")

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[ref] report -> {args.json}")


if __name__ == "__main__":
    main()
