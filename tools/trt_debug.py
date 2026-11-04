# -*- coding: utf-8 -*-
"""trt_debug.py —— 快速定位 Gather 语义问题（轴选择/模式）"""
import os
import sys
import numpy as np
import torch
import tensorrt as trt

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MD = os.path.join(PROJ, "botong25")
W, H, N = 5120, 4096, 5120 * 4096
PAD = W + 4
FLAT = N + PAD
LOGGER = trt.Logger(trt.Logger.WARNING)


def load_matrix(name):
    raw = np.fromfile(os.path.join(MD, name), dtype=np.float32)
    return raw.reshape(W, H).T


def main():
    # 用一张确定性的合成图，便于人工判断
    img = np.arange(N, dtype=np.uint32).astype(np.uint8).reshape(H, W)

    u0 = np.trunc(load_matrix("U_1.bin")).astype(np.int32)
    v0 = np.trunc(load_matrix("V_1.bin")).astype(np.int32)
    valid = ((u0 >= 1) & (v0 >= 1) & (u0 < W - 1) & (v0 < H - 1))
    idx0 = np.where(valid, v0.astype(np.int64) * W + u0, N).astype(np.int32)
    w = load_matrix("A1.bin").astype(np.float32)
    print("idx0 sample:", idx0.reshape(-1)[:5], "valid frac:", valid.mean())

    for axis in (0, 1):
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
        it = net.add_input("idx0", trt.int32, (1, N))
        g = net.add_gather(flat, it, axis)
        g.mode = trt.GatherMode.ELEMENT
        out = g.get_output(0)
        out.name = "out_f32"
        net.mark_output(out)

        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        ser = builder.build_serialized_network(net, cfg)
        eng = trt.Runtime(LOGGER).deserialize_cuda_engine(ser)
        ctx = eng.create_execution_context()
        stream = torch.cuda.current_stream().cuda_stream

        d_img = torch.from_numpy(img.reshape(-1)).to("cuda")
        d_idx = torch.from_numpy(idx0.reshape(-1)).to("cuda")
        d_out = torch.empty(N, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address("img", d_img.data_ptr())
        ctx.set_tensor_address("idx0", d_idx.data_ptr())
        ctx.set_tensor_address("out_f32", d_out.data_ptr())
        ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        got = d_out.cpu().numpy()

        flat_ref = np.concatenate([img.reshape(-1).astype(np.float32),
                                   np.zeros(PAD, np.float32)])
        exp = flat_ref[idx0.reshape(-1).astype(np.int64)]
        bad = int((got != exp).sum())
        print(f"axis={axis}: mismatches={bad}/{N}  sample got={got[:5]} exp={exp[:5]}")

        del d_img, d_idx, d_out, got
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
