# -*- coding: utf-8 -*-
"""
matrix_identify.py —— 用"越界黑边几何"判别两套矩阵集哪一套属于 pic 项目

线索来源：deployV1/pic/pic_http 文档多次提到
  「由于去畸变使得图像的四周多了大概 150 像素的黑边」
因此哪套矩阵产生的越界区域在几何上更接近"四周 ~150 px 黑边"，
哪套就更可能是 pic 线在用的矩阵集。
"""
import json
import os

import numpy as np

W, H, N = 5120, 4096, 5120 * 4096

SETS = {
    "botong25 (undistortion_standaloneV0)":
        r"E:\algrithm\project\botong\davissionNew\Algorithms\undistortion_standaloneV0\botong25",
    "chipBatchDetecting (Ref masks)":
        r"E:\algrithm\project\botong\davissionNew\Algorithms\Ref\projects\chipBatchDetectingQT\images\template\masks\removeDistortionMatrix",
}


def load_rowmajor(path):
    return np.fromfile(path, dtype=np.float32).reshape(W, H).T   # 列主序 -> 行主序 (H,W)


def analyze(tag, d):
    U = [load_rowmajor(os.path.join(d, f"U_{k+1}.bin")) for k in range(4)]
    V = [load_rowmajor(os.path.join(d, f"V_{k+1}.bin")) for k in range(4)]
    A = [load_rowmajor(os.path.join(d, f"A{k+1}.bin")) for k in range(4)]

    u = [np.trunc(x).astype(np.int32) for x in U]
    v = [np.trunc(x).astype(np.int32) for x in V]
    valid = np.ones((H, W), bool)
    for k in range(4):
        valid &= ((u[k] >= 1) & (v[k] >= 1) & (u[k] < W) & (v[k] < H))
    oob = ~valid

    # 越界区域的"等效边宽"：分别统计四条边上整列/整行是否全为越界
    def band(cols):
        return int(cols.sum())
    left = 0
    while left < W and oob[:, left].all():
        left += 1
    right = 0
    while right < W and oob[:, W - 1 - right].all():
        right += 1
    top = 0
    while top < H and oob[top, :].all():
        top += 1
    bottom = 0
    while bottom < H and oob[H - 1 - bottom, :].all():
        bottom += 1

    # 位移统计（仅有效像素）
    yy = np.arange(H, dtype=np.float64)[:, None]
    xx = np.arange(W, dtype=np.float64)[None, :]
    du = np.abs(U[0] - xx)[valid]
    dv = np.abs(V[0] - yy)[valid]

    # 权重和
    wsum = (A[0] + A[1] + A[2] + A[3])[valid].astype(np.float64)

    r = {
        "tag": tag, "dir": d,
        "oob_pixels": int(oob.sum()),
        "oob_ratio": float(oob.sum() / N),
        "full_band_px": {"left": left, "right": right, "top": top, "bottom": bottom},
        "abs_du": {"mean": float(du.mean()), "p99": float(np.percentile(du, 99)),
                   "max": float(du.max())},
        "abs_dv": {"mean": float(dv.mean()), "p99": float(np.percentile(dv, 99)),
                   "max": float(dv.max())},
        "weight_sum_abs_err_max": float(np.abs(wsum - 1.0).max()),
    }
    del U, V, A, u, v, valid, oob
    return r


def main():
    out = []
    for tag, d in SETS.items():
        print("=" * 70)
        print(tag)
        print(" ", d)
        r = analyze(tag, d)
        out.append(r)
        print(f"  越界像素      : {r['oob_pixels']:,}  ({r['oob_ratio']*100:.3f}%)")
        print(f"  整边全黑带宽度: left={r['full_band_px']['left']} right={r['full_band_px']['right']} "
              f"top={r['full_band_px']['top']} bottom={r['full_band_px']['bottom']}")
        print(f"  |Δu| mean/p99/max : {r['abs_du']['mean']:.2f} / "
              f"{r['abs_du']['p99']:.1f} / {r['abs_du']['max']:.0f}")
        print(f"  |Δv| mean/p99/max : {r['abs_dv']['mean']:.2f} / "
              f"{r['abs_dv']['p99']:.1f} / {r['abs_dv']['max']:.0f}")
        print(f"  权重和误差 max: {r['weight_sum_abs_err_max']:.3e}")

    p = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
                     "document", "data", "matrix_identify.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n->", p)


if __name__ == "__main__":
    main()
