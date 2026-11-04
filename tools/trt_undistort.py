# -*- coding: utf-8 -*-
"""
trt_undistort.py —— 去畸变的 TensorRT 原型（纯原生算子，无自定义 plugin）

四个实现分级，用于在"保证结果一致"的前提下给出最优方案与实测耗时：

  A     index4 + weight4 : 4 张 int32 索引表 + 4 张 fp32 权重表          (32 B/px)
  E     index1 + weight4 : 1 张 int32 基址表 + 3 次 +1/+W 派生 + 4 权重表 (20 B/px)  ← 推荐
  F     index1 + fu/fv   : 基址表 + fu/fv 两张表，权重由乘积重建         (12 B/px)
  FP16  E 的权重改 fp16  : 用于证明 fp16 不满足一致性要求

所有变体都与 C++ Release 产物 (demo/output0) 做逐像素比对。

用法：
  python tools/trt_undistort.py --variant E --images 3 --iters 30
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import tensorrt as trt

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MATRIX_DIR = os.path.join(PROJ, "botong25")
RAW = os.path.join(PROJ, "demo", "raw")
OUT = os.path.join(PROJ, "demo", "output0")
DATA = os.path.join(PROJ, "document", "data")

W, H, N = 5120, 4096, 5120 * 4096
PAD = W + 4          # 平坦输入尾部补零长度（覆盖 N..N+W+3 的哑元索引）
FLAT = N + PAD
LOGGER = trt.Logger(trt.Logger.WARNING)


# ----------------------------------------------------------------------
def load_matrix(name):
    raw = np.fromfile(os.path.join(MATRIX_DIR, name), dtype=np.float32)
    assert raw.size == N
    return raw.reshape(W, H).T          # (H, W) 行主序


def build_tables(variant):
    U = [np.trunc(load_matrix(f"U_{k+1}.bin")).astype(np.int32) for k in range(4)]
    V = [np.trunc(load_matrix(f"V_{k+1}.bin")).astype(np.int32) for k in range(4)]
    Wt = [load_matrix(f"A{k+1}.bin") for k in range(4)]

    valid = np.ones((H, W), bool)
    for k in range(4):
        valid &= ((U[k] >= 1) & (V[k] >= 1) & (U[k] < W) & (V[k] < H))

    idx = []
    for k in range(4):
        ik = V[k].astype(np.int64) * W + U[k].astype(np.int64)
        ik = np.where(valid, ik, N)
        idx.append(ik.astype(np.int32))

    out = {"n_valid": int(valid.sum())}
    if variant == "A":
        out["tables"] = {f"idx{k}": idx[k] for k in range(4)}
        out["tables"].update({f"w{k}": np.asarray(Wt[k], np.float32) for k in range(4)})
    elif variant in ("E", "FP16"):
        out["tables"] = {"idx0": idx[0]}
        np_dt = np.float16 if variant == "FP16" else np.float32
        out["tables"].update({f"w{k}": np.asarray(Wt[k], np_dt) for k in range(4)})
    elif variant == "F":
        fu = (Wt[1] + Wt[3]).astype(np.float32)
        fv = (Wt[2] + Wt[3]).astype(np.float32)
        out["tables"] = {"idx0": idx[0], "fu": fu, "fv": fv}
    else:
        raise ValueError(variant)
    return out


# ----------------------------------------------------------------------
def build_engine(variant):
    builder = trt.Builder(LOGGER)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    inp = net.add_input("img", trt.uint8, (1, 1, H, W))
    f32 = net.add_cast(inp, trt.float32)
    sh = net.add_shuffle(f32.get_output(0))
    sh.reshape_dims = (1, N)
    zeros = net.add_constant((1, PAD), trt.Weights(np.zeros(PAD, np.float32)))
    cat = net.add_concatenation([sh.get_output(0), zeros.get_output(0)])
    cat.axis = 1
    flat = cat.get_output(0)                     # [1, N+PAD] float32

    def gather_el(data, indices):
        lay = net.add_gather(data, indices, 1)
        lay.mode = trt.GatherMode.ELEMENT
        return lay

    if variant == "A":
        g, w_t = [], []
        for k in range(4):
            it = net.add_input(f"idx{k}", trt.int32, (1, N))
            g.append(gather_el(flat, it))
            w_t.append(net.add_input(f"w{k}", trt.float32, (1, N)))
    else:
        idx0 = net.add_input("idx0", trt.int32, (1, N))
        one = net.add_constant((1, 1), trt.Weights(np.array([1], np.int32)))
        woff = net.add_constant((1, 1), trt.Weights(np.array([W], np.int32)))
        d1 = net.add_elementwise(idx0, one.get_output(0), trt.ElementWiseOperation.SUM)
        d2 = net.add_elementwise(idx0, woff.get_output(0), trt.ElementWiseOperation.SUM)
        d3 = net.add_elementwise(d1.get_output(0), woff.get_output(0), trt.ElementWiseOperation.SUM)
        g = [gather_el(flat, idx0),
             gather_el(flat, d1.get_output(0)),
             gather_el(flat, d2.get_output(0)),
             gather_el(flat, d3.get_output(0))]
        if variant in ("E", "FP16"):
            if variant == "FP16":
                g = [net.add_cast(x.get_output(0), trt.float16) for x in g]
                w_t = [net.add_input(f"w{k}", trt.float16, (1, N)) for k in range(4)]
            else:
                w_t = [net.add_input(f"w{k}", trt.float32, (1, N)) for k in range(4)]
        else:  # F
            fu = net.add_input("fu", trt.float32, (1, N))
            fv = net.add_input("fv", trt.float32, (1, N))
            onef = net.add_constant((1, 1), trt.Weights(np.array([1.0], np.float32)))
            omfu = net.add_elementwise(onef.get_output(0), fu, trt.ElementWiseOperation.SUB)
            omfv = net.add_elementwise(onef.get_output(0), fv, trt.ElementWiseOperation.SUB)
            w_t = [net.add_elementwise(omfu.get_output(0), omfv.get_output(0),
                                       trt.ElementWiseOperation.PROD).get_output(0),
                   net.add_elementwise(fu, omfv.get_output(0),
                                       trt.ElementWiseOperation.PROD).get_output(0),
                   net.add_elementwise(omfu.get_output(0), fv,
                                       trt.ElementWiseOperation.PROD).get_output(0),
                   net.add_elementwise(fu, fv, trt.ElementWiseOperation.PROD).get_output(0)]

    g_t = [x.get_output(0) for x in g]
    acc = net.add_elementwise(g_t[0], w_t[0], trt.ElementWiseOperation.PROD)
    for k in (1, 2, 3):
        m = net.add_elementwise(g_t[k], w_t[k], trt.ElementWiseOperation.PROD)
        acc = net.add_elementwise(acc.get_output(0), m.get_output(0), trt.ElementWiseOperation.SUM)

    clip = net.add_activation(acc.get_output(0), trt.ActivationType.CLIP)
    clip.alpha = 0.0
    clip.beta = 255.0
    half = net.add_constant((1, 1), trt.Weights(np.array([0.5], np.float32)))
    addh = net.add_elementwise(clip.get_output(0), half.get_output(0), trt.ElementWiseOperation.SUM)
    fl = net.add_unary(addh.get_output(0), trt.UnaryOperation.FLOOR)
    u8 = net.add_cast(fl.get_output(0), trt.uint8)
    u8.get_output(0).name = "out_u8"
    net.mark_output(u8.get_output(0))

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    build_s = time.time() - t0
    if ser is None:
        raise RuntimeError("engine build failed")
    eng = trt.Runtime(LOGGER).deserialize_cuda_engine(ser)
    return eng, build_s, int(ser.nbytes)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="E", choices=["A", "E", "F", "FP16"])
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    import cv2

    print(f"[trt] variant={args.variant}: building tables ...")
    t0 = time.time()
    tb = build_tables(args.variant)
    print(f"[trt] tables ready in {time.time()-t0:.1f}s (valid px = {tb['n_valid']})")

    print("[trt] building engine ...")
    eng, build_s, eng_bytes = build_engine(args.variant)
    print(f"[trt] engine built in {build_s:.1f}s, serialized={eng_bytes/1e6:.1f} MB")

    ctx = eng.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream
    io_names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    print("[trt] IO:", io_names)

    dev = {}
    for nm in io_names:
        if nm in ("img", "out_u8"):
            continue
        arr = tb["tables"][nm]
        # 关键：load_matrix 返回的是转置视图（非连续），必须先 ascontiguousarray，
        # 否则 torch 会按 preserve_format 保留 strides，TRT 按行主序读取 -> 数据错位
        if arr.dtype == np.float16:
            flat = np.ascontiguousarray(arr, dtype=np.float16).reshape(-1)
            dev[nm] = torch.from_numpy(flat).to("cuda")
        elif arr.dtype == np.int32:
            flat = np.ascontiguousarray(arr, dtype=np.int32).reshape(-1)
            dev[nm] = torch.from_numpy(flat).to("cuda")
        else:
            flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
            dev[nm] = torch.from_numpy(flat).to("cuda")

    img_dev = torch.empty(N, dtype=torch.uint8, device="cuda")
    out_dev = torch.empty(N, dtype=torch.uint8, device="cuda")
    ctx.set_tensor_address("img", img_dev.data_ptr())
    ctx.set_tensor_address("out_u8", out_dev.data_ptr())
    for nm, t in dev.items():
        ctx.set_tensor_address(nm, t.data_ptr())

    names = sorted(os.listdir(RAW))[: args.images]
    results = []
    for nm in names:
        a = cv2.imread(os.path.join(RAW, nm), cv2.IMREAD_GRAYSCALE)
        ref = cv2.imread(os.path.join(OUT, nm), cv2.IMREAD_GRAYSCALE)
        img_dev.copy_(torch.from_numpy(a.reshape(-1)).to("cuda"))
        for _ in range(5):
            ctx.execute_async_v3(stream)
        torch.cuda.synchronize()

        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(args.iters):
            ctx.execute_async_v3(stream)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / args.iters

        got = out_dev.cpu().numpy().reshape(H, W)
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        nz = int((d != 0).sum())
        print(f"[trt] {nm}: {ms:8.3f} ms  diff={nz}  max={int(d.max())}  ({nz/d.size:.3e})")
        results.append({"name": nm, "gpu_ms": ms, "diff_pixels": nz,
                        "max_abs_diff": int(d.max()), "pixels": int(d.size)})

    report = {
        "variant": args.variant,
        "iters": args.iters,
        "engine_build_seconds": build_s,
        "engine_serialized_bytes": eng_bytes,
        "gpu": torch.cuda.get_device_name(0),
        "tensorrt": trt.__version__,
        "io_names": io_names,
        "table_bytes": {k: int(v.nbytes) for k, v in tb["tables"].items()},
        "table_bytes_total": int(sum(v.nbytes for v in tb["tables"].values())),
        "images": results,
        "gpu_ms_mean": float(np.mean([r["gpu_ms"] for r in results])),
        "total_diff_pixels": int(sum(r["diff_pixels"] for r in results)),
        "total_pixels": int(sum(r["pixels"] for r in results)),
    }
    print(f"\n[trt] mean GPU time = {report['gpu_ms_mean']:.3f} ms; "
          f"diff = {report['total_diff_pixels']}/{report['total_pixels']}")
    p = os.path.join(DATA, f"trt_prototype_{args.variant}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("[trt] ->", p)


if __name__ == "__main__":
    main()
