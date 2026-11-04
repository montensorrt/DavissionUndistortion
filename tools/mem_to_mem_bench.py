# -*- coding: utf-8 -*-
"""
mem_to_mem_bench.py —— 「单张图：主机内存 → 接口 → 主机内存」同口径时延对比

测量对象（同一张 5120x4096 灰度图，同一份算法语义）：

  CPU-P0    ImageUndistorter::infer(inBuf, W, H, outBuf)  —— 由 bench_undistortion.exe 测，
            该程序的 infer 区间就是「内存进 → 内存出」，不含任何文件 I/O。
  GPU-G1    pageable host  → H2D → kernel → D2H → sync   （串行，单 stream）
  GPU-G2    pinned   host  → H2D → kernel → D2H → sync   （串行，单 stream）
  GPU-G3    pinned + 3 stream 流水（H2D / kernel / D2H 重叠，双缓冲）→ 稳态每帧
  GPU-G4    kernel only（图已在显存，仅作参照）

所有 GPU 变体都用 TRT 变体 F（12 B/px），fp32，STRONGLY_TYPED。
"""
import argparse
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


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def build(variant="F"):
    tb = T.build_tables(variant)
    eng, build_s, eng_bytes = T.build_engine(variant)
    io = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    dev = {}
    for nm in io:
        if nm in ("img", "out_u8"):
            continue
        arr = tb["tables"][nm]
        dt = np.int32 if arr.dtype == np.int32 else np.float32
        dev[nm] = torch.from_numpy(np.ascontiguousarray(arr, dt).reshape(-1)).to("cuda")
    return eng, dev, io, build_s, int(sum(v.nbytes for v in tb["tables"].values()))


def make_ctx(eng, dev, d_in, d_out):
    ctx = eng.create_execution_context()
    ctx.set_tensor_address("img", d_in.data_ptr())
    ctx.set_tensor_address("out_u8", d_out.data_ptr())
    for nm, t in dev.items():
        ctx.set_tensor_address(nm, t.data_ptr())
    return ctx


def bench_serial(eng, dev, src, dst, pinned, iters):
    """单次完整调用的最小时延形式：async H2D → kernel → async D2H → 流同步。"""
    d_in = torch.empty(N, dtype=torch.uint8, device="cuda")
    d_out = torch.empty(N, dtype=torch.uint8, device="cuda")
    ctx = make_ctx(eng, dev, d_in, d_out)
    s = torch.cuda.current_stream().cuda_stream
    for _ in range(10):
        d_in.copy_(src, non_blocking=True)
        ctx.execute_async_v3(s)
        dst.copy_(d_out, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        d_in.copy_(src, non_blocking=True)
        ctx.execute_async_v3(s)
        dst.copy_(d_out, non_blocking=True)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / iters * 1000.0
    del d_in, d_out
    torch.cuda.empty_cache()
    return per


def bench_pipelined(eng, dev, src, dst, iters):
    """双缓冲 + 3 stream：H2D(i) / kernel(i) / D2H(i) 相互重叠，测稳态每帧。"""
    d_in = [torch.empty(N, dtype=torch.uint8, device="cuda") for _ in range(2)]
    d_out = [torch.empty(N, dtype=torch.uint8, device="cuda") for _ in range(2)]
    ctxs = [make_ctx(eng, dev, d_in[i], d_out[i]) for i in range(2)]
    s_in, s_ker, s_out = (torch.cuda.Stream() for _ in range(3))
    ev_in = [torch.cuda.Event() for _ in range(2)]
    ev_ker = [torch.cuda.Event() for _ in range(2)]

    def one(i):
        b = i & 1
        with torch.cuda.stream(s_in):
            d_in[b].copy_(src, non_blocking=True)
            ev_in[b].record(s_in)
        s_ker.wait_event(ev_in[b])
        with torch.cuda.stream(s_ker):
            ctxs[b].execute_async_v3(s_ker.cuda_stream)
            ev_ker[b].record(s_ker)
        s_out.wait_event(ev_ker[b])
        with torch.cuda.stream(s_out):
            dst.copy_(d_out[b], non_blocking=True)

    for i in range(10):
        one(i)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for i in range(iters):
        one(i)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / iters * 1000.0
    del d_in, d_out
    torch.cuda.empty_cache()
    return per


def bench_kernel_only(eng, dev, iters):
    d_in = torch.empty(N, dtype=torch.uint8, device="cuda")
    d_out = torch.empty(N, dtype=torch.uint8, device="cuda")
    ctx = make_ctx(eng, dev, d_in, d_out)
    s = torch.cuda.current_stream().cuda_stream
    for _ in range(10):
        ctx.execute_async_v3(s)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        ctx.execute_async_v3(s)
    e1.record()
    torch.cuda.synchronize()
    per = e0.elapsed_time(e1) / iters
    del d_in, d_out
    torch.cuda.empty_cache()
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--image", default="PIC_1_1.bmp")
    args = ap.parse_args()

    import cv2
    img = cv2.imread(os.path.join(RAW, args.image), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(os.path.join(OUT, args.image), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (H, W) and ref.shape == (H, W)

    pageable = torch.from_numpy(img.reshape(-1).copy())
    pinned_in = torch.empty(N, dtype=torch.uint8, pin_memory=True)
    pinned_in.copy_(pageable)
    pinned_out = torch.empty(N, dtype=torch.uint8, pin_memory=True)
    pageable_out = torch.empty(N, dtype=torch.uint8)

    print("[1/4] build TRT variant F engine ...")
    eng, dev, io, build_s, table_bytes = build("F")
    print(f"   engine build {build_s:.1f}s, tables {table_bytes/1e6:.1f} MB")

    r = {"image": args.image, "iters": args.iters,
         "table_bytes": table_bytes, "engine_build_seconds": build_s,
         "gpu": torch.cuda.get_device_name(0),
         "tensorrt": __import__("tensorrt").__version__,
         "width": W, "height": H, "pixels": N}

    print("[2/4] GPU G1 pageable serial ...")
    r["g1_pageable_serial_ms"] = bench_serial(eng, dev, pageable, pageable_out, False, args.iters)
    print(f"   {r['g1_pageable_serial_ms']:.3f} ms")

    print("[3/4] GPU G2 pinned serial ...")
    r["g2_pinned_serial_ms"] = bench_serial(eng, dev, pinned_in, pinned_out, True, args.iters)
    print(f"   {r['g2_pinned_serial_ms']:.3f} ms")

    print("[4/4] GPU G3 pinned pipelined ...")
    r["g3_pinned_pipelined_ms"] = bench_pipelined(eng, dev, pinned_in, pinned_out, args.iters)
    print(f"   {r['g3_pinned_pipelined_ms']:.3f} ms")

    r["g4_kernel_only_ms"] = bench_kernel_only(eng, dev, args.iters)
    print(f"   G4 kernel only {r['g4_kernel_only_ms']:.3f} ms")

    # 正确性（用 pinned 串行那一路的输出比对）
    d_in = torch.empty(N, dtype=torch.uint8, device="cuda")
    d_out = torch.empty(N, dtype=torch.uint8, device="cuda")
    ctx = make_ctx(eng, dev, d_in, d_out)
    d_in.copy_(pinned_in)
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    got = d_out.cpu().numpy().reshape(H, W)
    d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    r["diff_pixels"] = int((d != 0).sum())
    r["max_abs_diff"] = int(d.max())
    r["diff_ratio"] = r["diff_pixels"] / d.size
    print(f"   consistency: diff={r['diff_pixels']} max={r['max_abs_diff']} "
          f"ratio={r['diff_ratio']:.3e}")

    p = os.path.join(DATA, "mem_to_mem_bench.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print("\n->", p)


if __name__ == "__main__":
    main()
