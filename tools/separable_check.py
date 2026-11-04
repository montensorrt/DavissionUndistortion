# -*- coding: utf-8 -*-
"""
separable_check.py —— 严格验证映射场是否可分离 + A 权重能否由 frac 无损重建

若成立 => GPU 侧只需 u_lut[5120] + v_lut[4096]（约 36 KB），
          而不是 672 MB 的 4×int32 索引 + 4×fp32 权重。
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


def row(m, r):
    return m[:, r].copy()


U0, V0 = mm("U_1.bin"), mm("V_1.bin")
A = [mm(f"A{k+1}.bin") for k in range(4)]

rows = np.linspace(1, H - 2, 40).astype(np.int64)

# ---------- 有效像素掩码（与 C++ all-valid 语义一致，只需 tap0 已能判定） ----------
def valid_of(u0, v0):
    fx = np.trunc(u0).astype(np.int32)
    fy = np.trunc(v0).astype(np.int32)
    return (fx >= 1) & (fy >= 1) & (fx < W - 1) & (fy < H - 1)

# ---------- S1: u(x,y) 是否与 y 无关 ----------
base_r = int(rows[len(rows)//2])
u_base = row(U0, base_r)
v_base = row(V0, base_r)
valid_base = valid_of(u_base, v_base)

u_dev_max = 0.0
v_dev_max = 0.0
u_dev_px = 0
for r in rows:
    u = row(U0, r)
    v = row(V0, r)
    m = valid_base & valid_of(u, v)
    du = np.abs(u - u_base)[m]
    if du.size:
        u_dev_max = max(u_dev_max, float(du.max()))
        u_dev_px += int((du > 0).sum())

# v(x,y) 是否与 x 无关（同一行内 v 是否为常量）
v_inrow_max = 0.0
for r in rows:
    u = row(U0, r)
    v = row(V0, r)
    m = valid_of(u, v)
    vv = v[m]
    if vv.size:
        v_inrow_max = max(v_inrow_max, float(np.abs(vv - vv.mean()).max()))

print(f"[sep] u max deviation across rows (masked) = {u_dev_max}")
print(f"[sep] pixels where u differs from base row  = {u_dev_px}")
print(f"[sep] v max deviation within a row (masked) = {v_inrow_max}")

# ---------- S2: A 能否由 frac(u),frac(v) 无损重建 ----------
# 取整行的 A 与 u,v，逐像素比较
tot_px = 0
diff_px = 0
max_err = 0.0
for r in rows[:12]:
    u = row(U0, r)
    v = row(V0, r)
    a = [row(m, r) for m in A]
    fx = np.trunc(u)
    fy = np.trunc(v)
    fu = (u - fx).astype(np.float64)
    fv = (v - fy).astype(np.float64)
    recon = [(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv]
    m = valid_of(u, v)
    for k in range(4):
        e = np.abs(a[k].astype(np.float64) - recon[k])[m]
        if e.size:
            max_err = max(max_err, float(e.max()))
            diff_px += int((e > 0).sum())
    tot_px += int(m.sum())

print(f"[sep] A vs recomputed-from-frac: checked {tot_px} pixels x4 taps, "
      f"nonzero-diff entries={diff_px}, max_abs_err={max_err:.3e}")

# ---------- S3: 1D LUT 表示的一致性 ----------
# 用 base row 的 u 作为 u_lut(x)，用各行 v 的众数作为 v_lut(y)
u_lut = u_base.astype(np.float64)
v_lut = {}
for r in rows:
    u = row(U0, r)
    v = row(V0, r)
    m = valid_of(u, v)
    if m.sum():
        v_lut[int(r)] = float(np.median(v[m]))

# v_lut 是否平滑（相邻采样行的差值）
ks = sorted(v_lut.keys())
diffs = [v_lut[ks[i + 1]] - v_lut[ks[i]] for i in range(len(ks) - 1)]
print(f"[sep] v_lut range = [{min(v_lut.values()):.3f}, {max(v_lut.values()):.3f}], "
      f"row-step diff min={min(diffs):.3f} max={max(diffs):.3f}")
print(f"[sep] u_lut range = [{u_lut.min():.3f}, {u_lut.max():.3f}], "
      f"x-step diff min={np.diff(u_lut).min():.3f} max={np.diff(u_lut).max():.3f}")

report = {
    "u_max_dev_across_rows_masked": u_dev_max,
    "pixels_u_differs_across_rows": u_dev_px,
    "v_max_dev_within_row_masked": v_inrow_max,
    "a_recompute_checked_pixels": tot_px,
    "a_recompute_nonzero_diff_entries": diff_px,
    "a_recompute_max_abs_err": max_err,
    "v_lut_samples": {str(k): v for k, v in v_lut.items()},
    "u_lut_first": [float(x) for x in u_lut[:10]],
    "u_lut_last": [float(x) for x in u_lut[-10:]],
}
with open(os.path.join(DATA, "separable_check.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("[sep] ->", os.path.join(DATA, "separable_check.json"))
