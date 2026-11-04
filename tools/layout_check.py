# -*- coding: utf-8 -*-
"""
layout_check.py —— 判定 12 个 .bin 矩阵的真实内存布局与 (W,H) 语义

背景：CAPP/src/Undistortion.cpp 的 loadSimpleMatrix() 假定文件是
"列主序 (height 行长)"：rowMajor[row*W+col] = colMajor[row + col*H]。
若该假定与实际不符，去畸变输出会是"转置/旋转"的错误结果。

判别准则：正确布局下，映射应接近恒等（去畸变位移通常只有几十像素），
即 mean(|u - out_x|)、mean(|v - out_y|) 很小。
"""
import json
import os

import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATRIX_DIR = os.path.join(PROJ, "botong25")
DATA_DIR = os.path.join(PROJ, "document", "data")

W, H = 5120, 4096
N = W * H


def candidates(name):
    """返回 {描述: (H,W) 的 float32 视图}"""
    a = np.fromfile(os.path.join(MATRIX_DIR, name), dtype=np.float32)
    out = {}
    # 1) 文件即行主序，逻辑尺寸 (rows=H, cols=W)
    out["rowmajor_HxW"] = a.reshape(H, W)
    # 2) 文件即行主序，逻辑尺寸 (rows=W, cols=H)  -> 转置得到 (H,W)
    out["rowmajor_WxH_T"] = a.reshape(W, H).T
    # 3) 文件为列主序，逻辑尺寸 (W cols, H rows) —— 即 C++ 实现的假定
    out["colmajor_Wcols_T"] = a.reshape(W, H).T
    # 4) 文件为列主序，逻辑尺寸 (H cols, W rows)
    out["colmajor_Hcols_T"] = a.reshape(H, W).T
    return out


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    rows = np.linspace(0, H - 1, 32).astype(np.int64)
    cols = np.arange(0, W, 1, dtype=np.int64)

    u_c = candidates("U_1.bin")
    v_c = candidates("V_1.bin")

    print("Interpretation screening (using U_1/V_1):")
    print(f"{'layout':22s} {'mean|u-x|':>12s} {'max|u-x|':>12s} {'mean|v-y|':>12s} {'max|v-y|':>12s}")
    best = None
    results = {}
    for key in u_c:
        um = u_c[key]
        vm = v_c[key]
        # 采样
        u = um[np.ix_(rows, cols)].astype(np.float64)
        v = vm[np.ix_(rows, cols)].astype(np.float64)
        yy = np.repeat(rows.astype(np.float64)[:, None], W, axis=1)
        xx = np.repeat(cols.astype(np.float64)[None, :], rows.size, axis=0)
        du = np.abs(u - xx)
        dv = np.abs(v - yy)
        r = {"mean_du": float(du.mean()), "max_du": float(du.max()),
             "mean_dv": float(dv.mean()), "max_dv": float(dv.max())}
        results[key] = r
        print(f"{key:22s} {r['mean_du']:12.2f} {r['max_du']:12.2f} "
              f"{r['mean_dv']:12.2f} {r['max_dv']:12.2f}")
        score = r["mean_du"] + r["mean_dv"]
        if best is None or score < best[0]:
            best = (score, key)

    print(f"\n=> 最可能正确的布局: {best[1]}  (score={best[0]:.2f})")

    with open(os.path.join(DATA_DIR, "layout_check.json"), "w", encoding="utf-8") as f:
        json.dump({"results": results, "best_layout": best[1]}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
