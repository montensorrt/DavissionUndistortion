# -*- coding: utf-8 -*-
"""
map_probe.py —— 直接打印若干像素的 (u_i, v_i, a_i)，并统计映射的局部位移分布。
用于确认 4-tap 几何、A 权重的来源，以及 GPU 侧访存局部性。
"""
import os
import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD = os.path.join(PROJ, "botong25")
W, H, N = 5120, 4096, 5120 * 4096


def mm(name):
    return np.memmap(os.path.join(MD, name), dtype=np.float32, mode="r", shape=(W, H))


U = [mm(f"U_{k+1}.bin") for k in range(4)]
V = [mm(f"V_{k+1}.bin") for k in range(4)]
A = [mm(f"A{k+1}.bin") for k in range(4)]


def row(r):
    """取行主序第 r 行"""
    return [m[:, r].copy() for m in U], [m[:, r].copy() for m in V], [m[:, r].copy() for m in A]


print("=== 像素级样本（行主序 row=2000） ===")
u, v, a = row(2000)
for c in [0, 1, 2, 100, 2560, 4000, 5119]:
    print(f"col={c:5d}  u=({u[0][c]:9.3f},{u[1][c]:9.3f},{u[2][c]:9.3f},{u[3][c]:9.3f})"
          f"  v=({v[0][c]:9.3f},{v[1][c]:9.3f},{v[2][c]:9.3f},{v[3][c]:9.3f})")
    print(f"          a=({a[0][c]:9.6f},{a[1][c]:9.6f},{a[2][c]:9.6f},{a[3][c]:9.6f})")

print("\n=== tap 差分（全行，含越界像素） ===")
tu0 = np.trunc(u[0])
tv0 = np.trunc(v[0])
for k in (1, 2, 3):
    du = np.trunc(u[k]) - tu0
    dv = np.trunc(v[k]) - tv0
    print(f"tap{k}: du unique={np.unique(du)[:6]}  dv unique={np.unique(dv)[:6]}")

print("\n=== 有效像素子集上的位移分布 ===")
fx = np.trunc(u[0]).astype(np.int32)
fy = np.trunc(v[0]).astype(np.int32)
valid = (fx >= 1) & (fy >= 1) & (fx < W) & (fy < H)
xx = np.arange(W, dtype=np.float64)
du = np.abs(u[0].astype(np.float64) - xx)
dv = np.abs(v[0].astype(np.float64) - 2000.0)
print(f"valid pixels in row: {valid.sum()}/{W}")
for name, arr in (("|u-x|", du[valid]), ("|v-y|", dv[valid])):
    print(f"  {name}: mean={arr.mean():9.2f} p50={np.percentile(arr,50):9.2f} "
          f"p90={np.percentile(arr,90):9.2f} p99={np.percentile(arr,99):9.2f} max={arr.max():9.2f}")

print("\n=== frac 与 A 的关系（有效像素，试一试 4 种映射） ===")
fu = u[0].astype(np.float64) - fx
fv = v[0].astype(np.float64) - fy
cand = {
    "w00=(1-fu)(1-fv)": (1 - fu) * (1 - fv),
    "w01=fu(1-fv)": fu * (1 - fv),
    "w10=(1-fu)fv": (1 - fu) * fv,
    "w11=fu*fv": fu * fv,
}
for i in range(4):
    print(f"-- a{i} --")
    for name, c in cand.items():
        e = np.abs(a[i].astype(np.float64) - c)[valid]
        print(f"   {name:20s} mean={e.mean():.6f} max={e.max():.6f}")
