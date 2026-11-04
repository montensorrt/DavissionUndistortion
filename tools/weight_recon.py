# -*- coding: utf-8 -*-
"""
weight_recon.py —— 侧数据压缩可行性验证

问题：4 个 A 权重是否可无损压缩？
  2x2 双线性权重是秩 1 的：a0=(1-fu)(1-fv), a1=fu(1-fv), a2=(1-fu)fv, a3=fu*fv
  因此 fu = a1+a3, fv = a2+a3，只需 2 个 float 就能重建 4 个权重。

本脚本在真实数据上验证：
  R1 tap 几何是否固定为 (u0,v0),(u0+1,v0),(u0,v0+1),(u0+1,v0+1)
  R2 由 (a1+a3, a2+a3) 重建的 4 个权重与存储值是否逐字节一致（float32）
  R3 若用 int16 存 (u0,v0) + 2x fp32 存 (fu,fv)，侧数据可降到多少 B/px
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


U = [mm(f"U_{k+1}.bin") for k in range(4)]
V = [mm(f"V_{k+1}.bin") for k in range(4)]
A = [mm(f"A{k+1}.bin") for k in range(4)]

rows = np.linspace(2, H - 3, 24).astype(np.int64)

geo_bad = {"u1": 0, "v1": 0, "u2": 0, "v2": 0, "u3": 0, "v3": 0}
recon_bit_equal = [0, 0, 0, 0]
recon_total = 0
recon_max_err = 0.0
w_sum_max_err = 0.0
fu_range = [1e9, -1e9]
fv_range = [1e9, -1e9]

for r in rows:
    u = [row(m, r) for m in U]
    v = [row(m, r) for m in V]
    a = [row(m, r) for m in A]

    iu = [np.trunc(x).astype(np.int32) for x in u]
    iv = [np.trunc(x).astype(np.int32) for x in v]
    valid = ((iu[0] >= 1) & (iv[0] >= 1) & (iu[0] < W - 1) & (iv[0] < H - 1) &
             (iu[3] >= 1) & (iv[3] >= 1) & (iu[3] < W - 1) & (iv[3] < H - 1))
    if valid.sum() == 0:
        continue

    # R1 tap 几何
    geo_bad["u1"] += int(((iu[1] - iu[0]) != 1)[valid].sum())
    geo_bad["v1"] += int(((iv[1] - iv[0]) != 0)[valid].sum())
    geo_bad["u2"] += int(((iu[2] - iu[0]) != 0)[valid].sum())
    geo_bad["v2"] += int(((iv[2] - iv[0]) != 1)[valid].sum())
    geo_bad["u3"] += int(((iu[3] - iu[0]) != 1)[valid].sum())
    geo_bad["v3"] += int(((iv[3] - iv[0]) != 1)[valid].sum())

    # R2 权重重建
    a0, a1, a2, a3 = [x.astype(np.float32) for x in a]
    fu = (a1 + a3).astype(np.float32)
    fv = (a2 + a3).astype(np.float32)
    fu_range = [min(fu_range[0], float(fu[valid].min())), max(fu_range[1], float(fu[valid].max()))]
    fv_range = [min(fv_range[0], float(fv[valid].min())), max(fv_range[1], float(fv[valid].max()))]

    w0 = ((np.float32(1.0) - fu) * (np.float32(1.0) - fv)).astype(np.float32)
    w1 = (fu * (np.float32(1.0) - fv)).astype(np.float32)
    w2 = ((np.float32(1.0) - fu) * fv).astype(np.float32)
    w3 = (fu * fv).astype(np.float32)

    for k, (rec, orig) in enumerate(((w0, a0), (w1, a1), (w2, a2), (w3, a3))):
        recon_bit_equal[k] += int((rec[valid] == orig[valid]).sum())
        recon_max_err = max(recon_max_err, float(np.abs(rec[valid].astype(np.float64) -
                                                       orig[valid].astype(np.float64)).max()))
    s = (a0 + a1 + a2 + a3)[valid].astype(np.float64)
    w_sum_max_err = max(w_sum_max_err, float(np.abs(s - 1.0).max()))
    recon_total += int(valid.sum())

print(f"[recon] checked pixels: {recon_total}")
print(f"[recon] R1 tap-geometry violations: {geo_bad}")
print(f"[recon] R2 weight bit-equal counts: {recon_bit_equal} / {recon_total} each")
print(f"[recon] R2 max abs err of reconstructed weights = {recon_max_err:.3e}")
print(f"[recon] weight-sum max abs err = {w_sum_max_err:.3e}")
print(f"[recon] fu range = {fu_range}, fv range = {fv_range}")

report = {
    "checked_pixels": recon_total,
    "tap_geometry_violations": geo_bad,
    "weight_bit_equal_counts": recon_bit_equal,
    "weight_recon_max_abs_err": recon_max_err,
    "weight_sum_max_abs_err": w_sum_max_err,
    "fu_range": fu_range,
    "fv_range": fv_range,
    "bytes_per_pixel": {
        "current_4x_int32_plus_4x_fp32": 32,
        "u0v0_int16_plus_4x_fp32": 20,
        "u0v0_int16_plus_fu_fv_fp32": 12,
        "uv_fp32_pair": 8,
    },
}
with open(os.path.join(DATA, "weight_recon.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("[recon] ->", os.path.join(DATA, "weight_recon.json"))
