# -*- coding: utf-8 -*-
"""trt_debug2.py —— 逐级定位 E 变体的偏差（acc / clip / +0.5 / floor / cast）"""
import os
import numpy as np
import torch
import tensorrt as trt

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD = os.path.join(PROJ, "botong25")
W, H, N = 5120, 4096, 5120 * 4096
PAD = W + 4
LOGGER = trt.Logger(trt.Logger.WARNING)


def lm(name):
    return np.fromfile(os.path.join(MD, name), dtype=np.float32).reshape(W, H).T


def main():
    img = ((np.arange(N, dtype=np.uint64) * 7919) % 256).astype(np.uint8).reshape(H, W)

    U = [np.trunc(lm(f"U_{k+1}.bin")).astype(np.int32) for k in range(4)]
    V = [np.trunc(lm(f"V_{k+1}.bin")).astype(np.int32) for k in range(4)]
    Wt = [lm(f"A{k+1}.bin").astype(np.float32) for k in range(4)]
    valid = np.ones((H, W), bool)
    for k in range(4):
        valid &= ((U[k] >= 1) & (V[k] >= 1) & (U[k] < W) & (V[k] < H))
    idx0 = np.where(valid, V[0].astype(np.int64) * W + U[0], N).astype(np.int32)

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

    i0 = net.add_input("idx0", trt.int32, (1, N))
    one = net.add_constant((1, 1), trt.Weights(np.array([1], np.int32)))
    woff = net.add_constant((1, 1), trt.Weights(np.array([W], np.int32)))
    d1 = net.add_elementwise(i0, one.get_output(0), trt.ElementWiseOperation.SUM)
    d2 = net.add_elementwise(i0, woff.get_output(0), trt.ElementWiseOperation.SUM)
    d3 = net.add_elementwise(d1.get_output(0), woff.get_output(0), trt.ElementWiseOperation.SUM)
    g = [gel(flat, i0), gel(flat, d1.get_output(0)),
         gel(flat, d2.get_output(0)), gel(flat, d3.get_output(0))]
    wt = [net.add_input(f"w{k}", trt.float32, (1, N)) for k in range(4)]

    acc = net.add_elementwise(g[0].get_output(0), wt[0], trt.ElementWiseOperation.PROD)
    for k in (1, 2, 3):
        m = net.add_elementwise(g[k].get_output(0), wt[k], trt.ElementWiseOperation.PROD)
        acc = net.add_elementwise(acc.get_output(0), m.get_output(0), trt.ElementWiseOperation.SUM)

    outputs = {}
    acc.get_output(0).name = "acc"
    outputs["acc"] = (acc.get_output(0), trt.float32, N)

    clip = net.add_activation(acc.get_output(0), trt.ActivationType.CLIP)
    clip.alpha = 0.0
    clip.beta = 255.0
    clip.get_output(0).name = "clip"
    outputs["clip"] = (clip.get_output(0), trt.float32, N)

    half = net.add_constant((1, 1), trt.Weights(np.array([0.5], np.float32)))
    addh = net.add_elementwise(clip.get_output(0), half.get_output(0), trt.ElementWiseOperation.SUM)
    addh.get_output(0).name = "addh"
    outputs["addh"] = (addh.get_output(0), trt.float32, N)

    fl = net.add_unary(addh.get_output(0), trt.UnaryOperation.FLOOR)
    fl.get_output(0).name = "floor"
    outputs["floor"] = (fl.get_output(0), trt.float32, N)

    u8 = net.add_cast(fl.get_output(0), trt.uint8)
    u8.get_output(0).name = "out_u8"
    outputs["out_u8"] = (u8.get_output(0), trt.uint8, N)

    for nm, (t, _, _) in outputs.items():
        net.mark_output(t)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    ser = builder.build_serialized_network(net, cfg)
    eng = trt.Runtime(LOGGER).deserialize_cuda_engine(ser)
    ctx = eng.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream

    dev = {
        "img": torch.from_numpy(img.reshape(-1)).to("cuda"),
        "idx0": torch.from_numpy(idx0.reshape(-1)).to("cuda"),
        "w0": torch.from_numpy(np.ascontiguousarray(Wt[0]).reshape(-1)).to("cuda"),
        "w1": torch.from_numpy(np.ascontiguousarray(Wt[1]).reshape(-1)).to("cuda"),
        "w2": torch.from_numpy(np.ascontiguousarray(Wt[2]).reshape(-1)).to("cuda"),
        "w3": torch.from_numpy(np.ascontiguousarray(Wt[3]).reshape(-1)).to("cuda"),
    }
    out_dev = {}
    for nm, (t, dt, n) in outputs.items():
        out_dev[nm] = torch.empty(n, dtype=torch.uint8 if dt == trt.uint8 else torch.float32,
                                  device="cuda")
    for nm, t in dev.items():
        ctx.set_tensor_address(nm, t.data_ptr())
    for nm, t in out_dev.items():
        ctx.set_tensor_address(nm, t.data_ptr())
    ctx.execute_async_v3(stream)
    torch.cuda.synchronize()

    got = {nm: t.cpu().numpy() for nm, t in out_dev.items()}

    # ---- numpy 参考 ----
    flat_ref = np.concatenate([img.reshape(-1).astype(np.float32), np.zeros(PAD, np.float32)])
    ii = idx0.reshape(-1).astype(np.int64)
    s = [flat_ref[ii], flat_ref[ii + 1], flat_ref[ii + W], flat_ref[ii + W + 1]]
    w = [np.ascontiguousarray(Wt[k]).reshape(-1) for k in range(4)]
    ref_acc = s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + s[3] * w[3]
    ref_clip = np.clip(ref_acc, 0, 255)
    ref_addh = ref_clip + np.float32(0.5)
    ref_floor = np.floor(ref_addh)
    ref_u8 = ref_floor.astype(np.uint8)

    for nm, exp in (("acc", ref_acc), ("clip", ref_clip), ("addh", ref_addh),
                    ("floor", ref_floor), ("out_u8", ref_u8)):
        d = got[nm].astype(np.float64) - exp.astype(np.float64)
        nz = int((d != 0).sum())
        print(f"{nm:6s}: diff={nz:9d}  max|d|={np.abs(d).max():.6g}")

    # 逐 gather 检查
    for k in range(1, 4):
        pass
    # 详细诊断：取一个有效像素
    pos = 2000 * W + 2560
    print("\n--- diagnostic at (row=2000, col=2560) ---")
    print("idx0 =", ii[pos], " offset idx1/2/3 =", ii[pos]+1, ii[pos]+W, ii[pos]+W+1)
    print("s exp =", s[0][pos], s[1][pos], s[2][pos], s[3][pos])
    print("w exp =", w[0][pos], w[1][pos], w[2][pos], w[3][pos])
    print("acc exp =", ref_acc[pos])
    print("acc got =", got["acc"][pos])
    print("clip got=", got["clip"][pos], "addh got=", got["addh"][pos],
          "floor got=", got["floor"][pos], "u8 got=", got["out_u8"][pos])
    import numpy as _np
    d = _np.abs(got["acc"] - ref_acc)
    nzidx = _np.nonzero(d)[0][:10]
    print("first mismatching positions:", nzidx)
    for pp in nzidx[:5]:
        print(f"  pos={pp} idx0={ii[pp]} got={got['acc'][pp]} exp={ref_acc[pp]}")
    print("\nfirst row samples (acc):")
    print(" got", got["acc"][:6])
    print(" exp", ref_acc[:6])


if __name__ == "__main__":
    main()
