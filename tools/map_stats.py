# -*- coding: utf-8 -*-
"""
map_stats.py —— 全局统计映射场的结构特征

输出（供 TRT 方案设计与耗时预估使用）：
  S1 |u-x|、|v-y| 的分布（有效像素）
  S2 du/dx 的分布（相邻输出像素的源地址差）-> 决定 GPU gather 是否"近似合并访存"
  S3 可分离性：u 是否只与 x 有关、v 是否只与 y 有关
     若可分离，侧数据可从 20B/px 降到两个 1D LUT（~几十 KB），
     GPU 侧退化为"两次 1D 查表 + 一次 2x2 混合"，接近内存拷贝带宽
  S4 越界像素占比（边界语义复刻方式）
"""
import json
import os

import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD = os.path.join(PROJ, "botong25")
DATA = os.path.join(PROJ, "document", "data")
W, H, N = 5120, 4096, 5120 * 4096


def mm(name):
    return np.memmap(os.path.join(MD, name), dtype=np.float32, mode="r", shape=(W, H))


def getrow(m, r):
    return m[:, r].copy()


def main():
    os.makedirs(DATA, exist_ok=True)
    U = [mm(f"U_{k+1}.bin") for k in range(4)]
    V = [mm(f"V_{k+1}.bin") for k in range(4)]
    A = [mm(f"A{k+1}.bin") for k in range(4)]

    rows = np.linspace(1, H - 2, 48).astype(np.int64)

    du_all, dv_all, dudx_all, dvdy_all = [], [], [], []
    u_by_x = {}
    v_by_y = {}

    for r in rows:
        u0 = getrow(U[0], r)
        v0 = getrow(V[0], r)
        fx = np.trunc(u0).astype(np.int32)
        fy = np.trunc(v0).astype(np.int32)
        valid = (fx >= 1) & (fy >= 1) & (fx < W - 1) & (fy < H - 1)

        xx = np.arange(W, dtype=np.float64)
        du = np.abs(u0.astype(np.float64) - xx)[valid]
        dv = np.abs(v0.astype(np.float64) - r)[valid]
        du_all.append(du)
        dv_all.append(dv)

        # 相邻输出列之间的源坐标差（只在两端都有效的位置统计）
        both = valid[:-1] & valid[1:]
        dudx_all.append(np.abs(np.diff(u0.astype(np.float64)))[both])
        dvdy_all.append(np.abs(np.diff(v0.astype(np.float64)))[both])

        u_by_x[r] = u0.copy()
        v_by_y[r] = v0.copy()

    du = np.concatenate(du_all)
    dv = np.concatenate(dv_all)
    dudx = np.concatenate(dudx_all)
    dvdy = np.concatenate(dvdy_all)

    def q(a):
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
                "p999": float(np.percentile(a, 99.9)), "max": float(a.max())}

    stats = {"sampled_rows": int(rows.size), "sampled_pixels": int(rows.size * W)}
    stats["abs_du"] = q(du)
    stats["abs_dv"] = q(dv)
    stats["abs_du_dx"] = q(dudx)
    stats["abs_dv_dy"] = q(dvdy)

    # ---- 可分离性：u(x,y) 是否与 y 无关 ----
    ref_r = int(rows[len(rows) // 2])
    u_ref = u_by_x[ref_r]
    max_dev_u = 0.0
    for r in rows:
        max_dev_u = max(max_dev_u, float(np.abs(u_by_x[r] - u_ref).max()))
    stats["u_max_dev_across_rows"] = max_dev_u

    v_ref = v_by_y[ref_r]
    max_dev_v = 0.0
    for r in rows:
        # v(x,y) 是否与 x 无关：比较同一行的常量性
        vv = v_by_y[r]
        max_dev_v = max(max_dev_v, float(np.abs(vv - vv.mean()).max()))
    stats["v_max_dev_within_row"] = max_dev_v

    # ---- v 是否恒等于 y + 常数 ----
    stats["v_minus_y"] = []
    for r in rows[:8]:
        vv = v_by_y[r]
        stats["v_minus_y"].append({"row": int(r),
                                   "min": float((vv - r).min()),
                                   "max": float((vv - r).max())})

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    with open(os.path.join(DATA, "map_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("\n->", os.path.join(DATA, "map_stats.json"))


if __name__ == "__main__":
    main()
