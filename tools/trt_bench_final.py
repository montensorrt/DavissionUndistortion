# -*- coding: utf-8 -*-
"""
trt_bench_final.py —— TRT 原型的最终测量（用于设计文档引用）

测量项
  1) 三种表结构变体的端到端 GPU kernel 时延分布（p50/p90/mean/min）
  2) 逐像素一致性（对 C++ Release 产物 demo/output0）
  3) 显存带宽标定（大块 copy 的有效带宽）
  4) H2D / D2H 传输代价（21 MB 灰度图）
  5) fp16 权重的误差量级（用 numpy 解析，不依赖 TRT 是否能构建 fp16 引擎）
"""
import json
import os
import time

import numpy as np
import torch

import trt_undistort as T

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(PROJ, "demo", "raw")
OUT = os.path.join(PROJ, "demo", "output0")
DATA = os.path.join(PROJ, "document", "data")
W, H, N = 5120, 4096, 5120 * 4096


def pct(a, p):
    return float(np.percentile(np.asarray(a), p))


def measure_variant(variant, names, iters):
    import cv2
    tb = T.build_tables(variant)
    eng, build_s, eng_bytes = T.build_engine(variant)
    ctx = eng.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream
    io_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]

    dev = {}
    for nm in io_names:
        if nm in ("img", "out_u8"):
            continue
        arr = tb["tables"][nm]
        if arr.dtype == np.float16:
            flat = np.ascontiguousarray(arr, np.float16).reshape(-1)
        elif arr.dtype == np.int32:
            flat = np.ascontiguousarray(arr, np.int32).reshape(-1)
        else:
            flat = np.ascontiguousarray(arr, np.float32).reshape(-1)
        dev[nm] = torch.from_numpy(flat).to("cuda")

    img_dev = torch.empty(N, dtype=torch.uint8, device="cuda")
    out_dev = torch.empty(N, dtype=torch.uint8, device="cuda")
    ctx.set_tensor_address("img", img_dev.data_ptr())
    ctx.set_tensor_address("out_u8", out_dev.data_ptr())
    for nm, t in dev.items():
        ctx.set_tensor_address(nm, t.data_ptr())

    per_image = []
    total_diff = 0
    for nm in names:
        a = cv2.imread(os.path.join(RAW, nm), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(OUT, nm), cv2.IMREAD_GRAYSCALE)
        img_dev.copy_(torch.from_numpy(a.reshape(-1)))
        for _ in range(10):
            ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        for _ in range(iters):
            ctx.execute_async_v3(stream)
        torch.cuda.synchronize()

        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(iters):
            ctx.execute_async_v3(stream)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / iters

        got = out_dev.cpu().numpy().reshape(H, W)
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        nz = int((d != 0).sum())
        total_diff += nz
        per_image.append({"name": nm, "gpu_ms": ms, "diff_pixels": nz,
                          "max_abs_diff": int(d.max())})
        print(f"  [{variant}] {nm}: {ms:.4f} ms  diff={nz} max={int(d.max())}")

    ms_list = [x["gpu_ms"] for x in per_image]
    table_bytes = int(sum(v.nbytes for v in tb["tables"].values()))
    res = {
        "variant": variant,
        "iters": iters,
        "engine_build_seconds": build_s,
        "table_bytes": table_bytes,
        "table_bytes_per_pixel": table_bytes / N,
        "io_names": io_names,
        "torch_alloc_bytes": int(sum(t.element_size() * t.nelement() for t in dev.values())),
        "gpu_ms": {"mean": float(np.mean(ms_list)), "p50": pct(ms_list, 50),
                   "p90": pct(ms_list, 90), "min": float(np.min(ms_list)),
                   "max": float(np.max(ms_list))},
        "throughput_mpix_per_s": N / (pct(ms_list, 50) * 1e-3) / 1e6,
        "diff_pixels_total": total_diff,
        "diff_pixels_ratio": total_diff / (N * len(names)),
        "per_image": per_image,
    }
    del dev, img_dev, out_dev, eng, ctx
    torch.cuda.empty_cache()
    return res


def measure_bandwidth():
    n = 256 * 1024 * 1024 // 4          # 256 MB 源
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(5):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    iters = 30
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / iters
    # copy = 读 + 写
    gbs = (2 * n * 4) / (ms * 1e-3) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return {"copy_ms": ms, "copy_bytes_one_way": n * 4, "effective_gbs": gbs}


def measure_transfer(names):
    import cv2
    img = cv2.imread(os.path.join(RAW, names[0]), cv2.IMREAD_GRAYSCALE).reshape(-1)
    src = torch.from_numpy(img.copy())
    dst = torch.empty(N, dtype=torch.uint8, device="cuda")
    out = torch.empty(N, dtype=torch.uint8)

    for _ in range(5):
        dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        dst.copy_(src, non_blocking=False)
    torch.cuda.synchronize()
    h2d = (time.perf_counter() - t0) / 20 * 1000

    for _ in range(5):
        out.copy_(dst, non_blocking=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        out.copy_(dst, non_blocking=False)
    torch.cuda.synchronize()
    d2h = (time.perf_counter() - t0) / 20 * 1000

    return {"bytes": N, "h2d_ms": h2d, "d2h_ms": d2h,
            "h2d_gbs": N / (h2d * 1e-3) / 1e9, "d2h_gbs": N / (d2h * 1e-3) / 1e9}


def analyse_fp16():
    """把权重降到 fp16 后的量化/累计误差（解析估计 + 抽样实测）"""
    import cv2
    Wt = [np.ascontiguousarray(T.load_matrix(f"A{k+1}.bin"), np.float32).reshape(-1)
          for k in range(4)]
    U = [np.trunc(T.load_matrix(f"U_{k+1}.bin")).astype(np.int32) for k in range(4)]
    V = [np.trunc(T.load_matrix(f"V_{k+1}.bin")).astype(np.int32) for k in range(4)]
    valid = np.ones((H, W), bool)
    for k in range(4):
        valid &= ((U[k] >= 1) & (V[k] >= 1) & (U[k] < W) & (V[k] < H))
    idx0 = np.where(valid, V[0].astype(np.int64) * W + U[0], N).astype(np.int32).reshape(-1)

    # 抽样 200 万像素
    rng = np.random.default_rng(0)
    sel = rng.choice(idx0.size, 2_000_000, replace=False)
    img = cv2.imread(os.path.join(RAW, "PIC_1_1.bmp"), cv2.IMREAD_GRAYSCALE).reshape(-1)
    flat = np.concatenate([img.astype(np.float32), np.zeros(W + 4, np.float32)])
    ii = idx0[sel].astype(np.int64)
    s = [flat[ii], flat[ii + 1], flat[ii + W], flat[ii + W + 1]]
    w32 = [Wt[k][sel] for k in range(4)]
    acc32 = s[0] * w32[0] + s[1] * w32[1] + s[2] * w32[2] + s[3] * w32[3]
    q32 = np.trunc(np.clip(acc32, 0, 255) + np.float32(0.5)).astype(np.uint8)

    w16 = [Wt[k][sel].astype(np.float16) for k in range(4)]
    acc16 = (s[0].astype(np.float16) * w16[0] + s[1].astype(np.float16) * w16[1]
             + s[2].astype(np.float16) * w16[2] + s[3].astype(np.float16) * w16[3])
    q16 = np.trunc(np.clip(acc16.astype(np.float32), 0, 255) + np.float32(0.5)).astype(np.uint8)

    d = np.abs(q16.astype(np.int16) - q32.astype(np.int16))
    return {"sampled_pixels": int(sel.size),
            "fp16_diff_pixels": int((d != 0).sum()),
            "fp16_diff_ratio": float((d != 0).sum() / sel.size),
            "fp16_max_abs_diff": int(d.max()),
            "acc_float16_max_abs_err": float(np.abs(acc16.astype(np.float64)
                                                    - acc32.astype(np.float64)).max())}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--variants", default="A,E,F")
    args = ap.parse_args()

    names = sorted(os.listdir(RAW))[: args.images]
    report = {"gpu": torch.cuda.get_device_name(0),
              "tensorrt": __import__("tensorrt").__version__,
              "torch": torch.__version__,
              "images": names, "iters": args.iters}

    print("[1/5] bandwidth calibration ...")
    report["bandwidth"] = measure_bandwidth()
    print("   ", report["bandwidth"])

    print("[2/5] H2D/D2H transfer ...")
    report["transfer"] = measure_transfer(names)
    print("   ", report["transfer"])

    print("[3/5] fp16 sensitivity ...")
    report["fp16"] = analyse_fp16()
    print("   ", report["fp16"])

    print("[4/5] variants ...")
    report["variants"] = {}
    for v in args.variants.split(","):
        v = v.strip()
        if not v:
            continue
        print(f" -- variant {v} --")
        report["variants"][v] = measure_variant(v, names, args.iters)

    print("[5/5] summary")
    for v, r in report["variants"].items():
        g = r["gpu_ms"]
        print(f"  {v}: p50={g['p50']:.4f} ms mean={g['mean']:.4f} ms "
              f"table={r['table_bytes_per_pixel']:.1f} B/px "
              f"({r['table_bytes']/1e6:.1f} MB) "
              f"diff={r['diff_pixels_ratio']:.3e}")

    p = os.path.join(DATA, "trt_bench_final.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("\n->", p)


if __name__ == "__main__":
    main()
