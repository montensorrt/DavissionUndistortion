# -*- coding: utf-8 -*-
"""
io_decompose.py —— 拆解「端到端 103 ms」里与算法无关的那部分

bench_undistortion 实测（40 图 × 8 次，Release）：
    read 22.725 + copy 1.099 + infer 63.817 + write 15.415 = 103.056 ms

本脚本把 read / write 再拆成「BMP 编解码」与「磁盘读写」两部分：
    read  = 磁盘/页缓存读 + BMP 解码
    write = BMP 编码 + 磁盘写
方法：分别测 open().read()、cv2.imdecode()、cv2.imread()、cv2.imwrite()。
"""
import json
import os
import time

import numpy as np
import cv2

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(PROJ, "demo", "raw")
DATA = os.path.join(PROJ, "document", "data")
NAME = "PIC_1_1.bmp"
PATH = os.path.join(RAW, NAME)

REPS = 12


def timeit(fn, reps=REPS, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    return {"mean": float(ts.mean()), "p50": float(np.percentile(ts, 50)),
            "min": float(ts.min()), "max": float(ts.max()), "n": int(ts.size)}


def main():
    os.makedirs(DATA, exist_ok=True)
    size = os.path.getsize(PATH)

    # 1) 纯文件读（页缓存已热）
    buf_holder = {}

    def raw_read():
        with open(PATH, "rb") as f:
            buf_holder["b"] = f.read()

    # 2) 纯 BMP 解码（输入已在内存）
    raw_read()
    arr = np.frombuffer(buf_holder["b"], np.uint8)

    def decode_only():
        cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    # 3) cv2.imread（读 + 解码）
    def imread_full():
        cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)

    # 4) 编码 + 写
    img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    tmp_out = os.path.join(PROJ, "demo", "_io_probe.bmp")

    def imwrite_full():
        cv2.imwrite(tmp_out, img)

    # 5) 仅编码（imencode，不落盘）
    def encode_only():
        cv2.imencode(".bmp", img)

    r = {}
    r["file_bytes"] = size
    r["raw_read"] = timeit(raw_read)
    r["decode_only"] = timeit(decode_only)
    r["imread_full"] = timeit(imread_full)
    r["encode_only"] = timeit(encode_only)
    r["imwrite_full"] = timeit(imwrite_full)

    # 推演
    r["read_total_ms"] = r["imread_full"]["p50"]
    r["read_decode_ms"] = r["decode_only"]["p50"]
    r["read_disk_ms"] = max(0.0, r["read_total_ms"] - r["read_decode_ms"])
    r["write_total_ms"] = r["imwrite_full"]["p50"]
    r["write_encode_ms"] = r["encode_only"]["p50"]
    r["write_disk_ms"] = max(0.0, r["write_total_ms"] - r["write_encode_ms"])

    # 与 bench 实测对齐（40 图均值）
    bench = {"read": 22.725, "copy": 1.099, "infer": 63.817, "write": 15.415}
    bench["e2e"] = sum(bench[k] for k in ("read", "copy", "infer", "write"))
    r["bench"] = bench
    r["bench"]["io_total"] = bench["read"] + bench["copy"] + bench["write"]
    r["bench"]["io_ratio"] = r["bench"]["io_total"] / bench["e2e"]
    r["bench"]["algo_ratio"] = bench["infer"] / bench["e2e"]

    if os.path.exists(tmp_out):
        os.remove(tmp_out)

    for k in ("raw_read", "decode_only", "imread_full", "encode_only", "imwrite_full"):
        v = r[k]
        print(f"{k:14s} p50={v['p50']:7.3f} ms  mean={v['mean']:7.3f} "
              f"min={v['min']:7.3f} max={v['max']:7.3f}")
    print()
    print(f"read  拆解: 磁盘/页缓存 {r['read_disk_ms']:6.2f} + BMP 解码 {r['read_decode_ms']:6.2f} "
          f"= {r['read_total_ms']:6.2f} ms")
    print(f"write 拆解: BMP 编码 {r['write_encode_ms']:6.2f} + 落盘 {r['write_disk_ms']:6.2f} "
          f"= {r['write_total_ms']:6.2f} ms")
    print()
    print(f"bench e2e = {bench['e2e']:.3f} ms")
    print(f"  算法 infer      = {bench['infer']:.3f} ms  ({bench['algo_ratio']*100:.1f}%)")
    print(f"  非算法 I/O 合计 = {bench['io_total']:.3f} ms  ({bench['io_ratio']*100:.1f}%)")

    p = os.path.join(DATA, "io_decompose.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print("\n->", p)


if __name__ == "__main__":
    main()
