# -*- coding: utf-8 -*-
"""
exactness_audit.py —— 「TRT 结果 vs CPU 结果」的双域逐位置审计

回答一个可验收的问题：**每个位置的差异到底有多大？**

两个域各看一遍：
  ① 浮点域（量化前，pre-clip acc）：|Δ| 的最大值 / 分布
     —— 这正是"差异低于 0.001 灰度"这句话能被严格验证的域
  ② 灰度域（量化后 uint8）：差 0 / 差 1 / 差 ≥2 的像素数，以及等效 MAE

基准：CPU Release 产物
  - uint8 基准：demo/output0/*.bmp（ImageUndistorter::infer 直接写出）
  - float 基准：document/data/cpu_float/*.bmp.f32
                （由 bench_undistortion --dump-float 导出，与库内 tempBuffer_ 逐位相同）

用法：
  python tools/exactness_audit.py --images 3
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import tensorrt as trt

import trt_undistort as T

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(PROJ, "demo", "raw")
OUT = os.path.join(PROJ, "demo", "output0")
FLT = os.path.join(PROJ, "document", "data", "cpu_float")
DATA = os.path.join(PROJ, "document", "data")
W, H, N = 5120, 4096, 5120 * 4096
PAD = W + 4
LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine_two_outputs():
    """与变体 F 相同的图，但同时暴露 pre-clip acc 与最终 uint8。"""
    builder = trt.Builder(LOGGER)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    inp = net.add_input("img", trt.uint8, (1, 1, H, W))
    f32 = net.add_cast(inp, trt.float32)
    sh = net.add_shuffle(f32.get_output(0))
    sh.reshape_dims = (1, N)
    zeros = net.add_constant((1, PAD), trt.Weights(np.zeros(PAD, np.float32)))
    cat = net.add_concatenation([sh.get_output(0), zeros.get_output(0)])
    cat.axis = 1
    flat = cat.get_output(0)

    def gel(data, ind):
        lay = net.add_gather(data, ind, 1)
        lay.mode = trt.GatherMode.ELEMENT
        return lay

    idx0 = net.add_input("idx0", trt.int32, (1, N))
    one = net.add_constant((1, 1), trt.Weights(np.array([1], np.int32)))
    woff = net.add_constant((1, 1), trt.Weights(np.array([W], np.int32)))
    d1 = net.add_elementwise(idx0, one.get_output(0), trt.ElementWiseOperation.SUM)
    d2 = net.add_elementwise(idx0, woff.get_output(0), trt.ElementWiseOperation.SUM)
    d3 = net.add_elementwise(d1.get_output(0), woff.get_output(0), trt.ElementWiseOperation.SUM)
    g = [gel(flat, idx0), gel(flat, d1.get_output(0)),
         gel(flat, d2.get_output(0)), gel(flat, d3.get_output(0))]

    fu = net.add_input("fu", trt.float32, (1, N))
    fv = net.add_input("fv", trt.float32, (1, N))
    onef = net.add_constant((1, 1), trt.Weights(np.array([1.0], np.float32)))
    omfu = net.add_elementwise(onef.get_output(0), fu, trt.ElementWiseOperation.SUB)
    omfv = net.add_elementwise(onef.get_output(0), fv, trt.ElementWiseOperation.SUB)
    w = [net.add_elementwise(omfu.get_output(0), omfv.get_output(0),
                             trt.ElementWiseOperation.PROD).get_output(0),
         net.add_elementwise(fu, omfv.get_output(0),
                             trt.ElementWiseOperation.PROD).get_output(0),
         net.add_elementwise(omfu.get_output(0), fv,
                             trt.ElementWiseOperation.PROD).get_output(0),
         net.add_elementwise(fu, fv, trt.ElementWiseOperation.PROD).get_output(0)]
    gt = [x.get_output(0) for x in g]

    acc = net.add_elementwise(gt[0], w[0], trt.ElementWiseOperation.PROD)
    for k in (1, 2, 3):
        m = net.add_elementwise(gt[k], w[k], trt.ElementWiseOperation.PROD)
        acc = net.add_elementwise(acc.get_output(0), m.get_output(0),
                                  trt.ElementWiseOperation.SUM)
    acc.get_output(0).name = "acc_f32"
    net.mark_output(acc.get_output(0))

    clip = net.add_activation(acc.get_output(0), trt.ActivationType.CLIP)
    clip.alpha = 0.0
    clip.beta = 255.0
    half = net.add_constant((1, 1), trt.Weights(np.array([0.5], np.float32)))
    addh = net.add_elementwise(clip.get_output(0), half.get_output(0),
                               trt.ElementWiseOperation.SUM)
    fl = net.add_unary(addh.get_output(0), trt.UnaryOperation.FLOOR)
    u8 = net.add_cast(fl.get_output(0), trt.uint8)
    u8.get_output(0).name = "out_u8"
    net.mark_output(u8.get_output(0))

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    build_s = time.time() - t0
    eng = trt.Runtime(LOGGER).deserialize_cuda_engine(ser)
    return eng, build_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--thr", type=float, default=1e-3,
                    help="阈值（灰度级），用于统计 |Δ| 超过它的位置数")
    args = ap.parse_args()

    import cv2

    names = sorted(os.listdir(RAW))[: args.images]
    print(f"[audit] images = {names}")

    print("[audit] building tables + engine ...")
    tb = T.build_tables("F")
    eng, build_s = build_engine_two_outputs()
    print(f"[audit] engine built in {build_s:.1f}s")

    ctx = eng.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream

    io = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    dev = {}
    for nm in io:
        if nm in ("img", "acc_f32", "out_u8"):
            continue
        arr = tb["tables"][nm]
        dt = np.int32 if arr.dtype == np.int32 else np.float32
        dev[nm] = torch.from_numpy(np.ascontiguousarray(arr, dt).reshape(-1)).to("cuda")
    d_in = torch.empty(N, dtype=torch.uint8, device="cuda")
    d_acc = torch.empty(N, dtype=torch.float32, device="cuda")
    d_out = torch.empty(N, dtype=torch.uint8, device="cuda")

    ctx.set_tensor_address("img", d_in.data_ptr())
    ctx.set_tensor_address("acc_f32", d_acc.data_ptr())
    ctx.set_tensor_address("out_u8", d_out.data_ptr())
    for nm, t in dev.items():
        ctx.set_tensor_address(nm, t.data_ptr())

    thr = args.thr
    per_image = []
    for nm in names:
        img = cv2.imread(os.path.join(RAW, nm), cv2.IMREAD_GRAYSCALE)
        ref_u8 = cv2.imread(os.path.join(OUT, nm), cv2.IMREAD_GRAYSCALE)
        fpath = os.path.join(FLT, nm + ".f32")
        ref_f = np.fromfile(fpath, dtype=np.float32)
        assert ref_f.size == N, (fpath, ref_f.size)

        d_in.copy_(torch.from_numpy(img.reshape(-1)))
        ctx.execute_async_v3(stream)
        torch.cuda.synchronize()

        got_f = d_acc.cpu().numpy()
        got_u8 = d_out.cpu().numpy().reshape(H, W)

        # ① 浮点域
        df = np.abs(got_f.astype(np.float64) - ref_f.astype(np.float64))
        fmax = float(df.max())
        fmean = float(df.mean())
        frmse = float(np.sqrt((df * df).mean()))
        f_over = {
            "gt_1e-3": int((df > 1e-3).sum()),
            "gt_1e-4": int((df > 1e-4).sum()),
            "gt_1e-5": int((df > 1e-5).sum()),
            "gt_1e-6": int((df > 1e-6).sum()),
            "gt_0": int((df > 0).sum()),
        }
        fq = {p: float(np.percentile(df, p)) for p in (50, 90, 99, 99.9, 99.99, 99.999)}

        # ② 灰度域
        du = np.abs(got_u8.astype(np.int16) - ref_u8.astype(np.int16))
        nz = int((du != 0).sum())
        u = {
            "diff_pixels": nz,
            "diff_ratio": nz / du.size,
            "max_abs_diff": int(du.max()),
            "count_eq_1": int((du == 1).sum()),
            "count_ge_2": int((du >= 2).sum()),
            "mae_gray": float(du.mean()),
            "rmse_gray": float(np.sqrt((du.astype(np.float64) ** 2).mean())),
            "psnr_db": None,
        }
        if u["rmse_gray"] > 0:
            u["psnr_db"] = float(20 * np.log10(255.0 / u["rmse_gray"]))

        print(f"\n[audit] {nm}")
        print(f"  float  |Δ|max = {fmax:.3e}  mean = {fmean:.3e}  rmse = {frmse:.3e}")
        print(f"         >1e-3: {f_over['gt_1e-3']}   >1e-4: {f_over['gt_1e-4']}   "
              f">1e-5: {f_over['gt_1e-5']}   >0: {f_over['gt_0']}")
        print(f"         p99.99 = {fq[99.99]:.3e}  p99.999 = {fq[99.999]:.3e}")
        print(f"  uint8  diff = {nz}/{du.size} = {u['diff_ratio']:.3e}  "
              f"max = {u['max_abs_diff']}  (=1: {u['count_eq_1']}, >=2: {u['count_ge_2']})")
        print(f"         MAE = {u['mae_gray']:.3e} gray  "
              f"RMSE = {u['rmse_gray']:.3e} gray  "
              f"PSNR = {u['psnr_db'] if u['psnr_db'] else float('inf'):.2f} dB")

        per_image.append({
            "name": nm,
            "float": {"max_abs": fmax, "mean_abs": fmean, "rmse": frmse,
                      "over": f_over, "percentiles": fq},
            "uint8": u,
        })

    # 汇总
    fmax_all = max(x["float"]["max_abs"] for x in per_image)
    fmean_all = float(np.mean([x["float"]["mean_abs"] for x in per_image]))
    over_all = sum(x["float"]["over"]["gt_1e-3"] for x in per_image)
    nz_all = sum(x["uint8"]["diff_pixels"] for x in per_image)
    tot = sum(x["uint8"]["pixels"] if "pixels" in x["uint8"] else N for x in per_image)
    maxdiff_all = max(x["uint8"]["max_abs_diff"] for x in per_image)
    mae_all = float(np.mean([x["uint8"]["mae_gray"] for x in per_image]))

    print("\n" + "=" * 74)
    print(f"[audit] 汇总（{len(names)} 张图 × {N} 像素 = {N*len(names):,} 个位置）")
    print(f"  ① 浮点域 |Δ| 最大值            = {fmax_all:.3e} 灰度")
    print(f"     浮点域 |Δ| 平均值            = {fmean_all:.3e} 灰度")
    print(f"     |Δ| > {thr:g} 的位置数        = {over_all}  "
          f"{'<-- 全部为 0，严格满足 < 0.001' if over_all == 0 else ''}")
    print(f"  ② 灰度域 差异像素              = {nz_all:,} / {tot:,} = {nz_all/tot:.3e}")
    print(f"     灰度域 最大差                = {maxdiff_all} 灰度级")
    print(f"     灰度域 等效 MAE              = {mae_all:.3e} 灰度级")
    print("=" * 74)

    rep = {
        "images": names, "pixels_per_image": N, "total_positions": tot,
        "threshold": thr,
        "float": {"max_abs_all": fmax_all, "mean_abs_all": fmean_all,
                  "positions_over_threshold": over_all},
        "uint8": {"diff_pixels_all": nz_all, "diff_ratio_all": nz_all / tot,
                  "max_abs_all": maxdiff_all, "mae_gray_all": mae_all},
        "per_image": per_image,
    }
    p = os.path.join(DATA, "exactness_audit.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("->", p)


if __name__ == "__main__":
    main()
