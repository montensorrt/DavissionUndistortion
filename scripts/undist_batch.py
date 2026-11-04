#!/usr/bin/env python3
"""批量去畸变（TensorRT，Algorithms/undistortion）：按子目录批量处理 → 同名同结构输出。

输入 : <raw>/<case>/<name>.bmp    (5120x4096 单通道 uint8)
输出 : <undis>/<case>/<name>.bmp  (同尺寸 8 位 BMP，文件名不变)

去畸变矩阵：Algorithms/undistortion_standaloneV0/botong25（U_1..4 / V_1..4 / A1..4，12 x 80 MiB）
走 pinned 零拷贝（cudaHostAlloc 由 cudart 直接分配）；不带 pinned 时自动退化到 pageable。

用法（仓库根目录）：
  python Algorithms/undistortion/scripts/undist_batch.py \
    --raw  dataset/davisson/cpu16/pic_cpo16/ee2-pic/raw \
    --out  dataset/davisson/cpu16/pic_cpo16/ee2-pic/undis

  # 自检：把 recipe/raw/pic 去畸变后与 recipe/test/pic（已是去畸变结果）逐像素对比
  python Algorithms/undistortion/scripts/undist_batch.py --selfcheck
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DLL = ROOT / "Algorithms/undistortion/trt/build/undist_api.dll"
MATRIX_DIR = ROOT / "Algorithms/undistortion_standaloneV0/botong25"
CACHE = ROOT / "Algorithms/undistortion/trt/build/cache"
UNDIST_OK = 0


# --------------------------------------------------------------------- C ABI
class UndistOptions(C.Structure):
    _fields_ = [("memory_mode", C.c_int32), ("device_id", C.c_int32),
                ("channels", C.c_int32), ("log_level", C.c_int32)]


class UndistInfo(C.Structure):
    _fields_ = [
        ("table_bytes", C.c_int64), ("engine_bytes", C.c_int64), ("device_bytes", C.c_int64),
        ("table_bytes_per_pixel", C.c_double),
        ("init_table_ms", C.c_double), ("init_engine_ms", C.c_double), ("init_total_ms", C.c_double),
        ("width", C.c_int32), ("height", C.c_int32), ("channels", C.c_int32),
        ("memory_mode", C.c_int32), ("table_variant", C.c_int32), ("sm_count", C.c_int32),
        ("gpu_name", C.c_char * 96), ("tensorrt_version", C.c_char * 32),
    ]


class UndistTiming(C.Structure):
    _fields_ = [("h2d_ms", C.c_double), ("infer_ms", C.c_double),
                ("d2h_ms", C.c_double), ("total_ms", C.c_double)]


def prepare_dll_dirs() -> list[str]:
    cands: list[Path] = []
    for env in ("TRT_ROOT", "TENSORRT_ROOT", "CUDA_PATH", "CUDA_HOME"):
        v = os.environ.get(env)
        if v:
            cands.append(Path(v))
    for root in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT",
                 r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"):
        cands.append(Path(root))
    dirs: list[str] = []
    for base in cands:
        if not base.exists():
            continue
        if any(base.glob("nvinfer_10.dll")) or any(base.glob("cudart64_12.dll")):
            dirs.append(str(base))
        for sub in sorted(base.glob("*/bin")):
            dirs.append(str(sub))
            dirs += [str(x) for x in sorted(sub.glob("*/bin"))]
    seen, out = set(), []
    for d in dirs:
        if d not in seen and Path(d).exists():
            seen.add(d)
            out.append(d)
    for d in out:
        try:
            os.add_dll_directory(d)
        except (AttributeError, OSError):
            pass
    os.environ["PATH"] = ";".join(out + [os.environ.get("PATH", "")])
    return out


class Cudart:
    """只用 cudaHostAlloc / cudaFreeHost 做真正的页锁缓冲（可选）"""

    def __init__(self) -> None:
        self.lib = None
        for name in ("cudart64_12.dll", "cudart64_13.dll"):
            try:
                lib = C.CDLL(name)
                lib.cudaHostAlloc.restype = C.c_int32
                lib.cudaHostAlloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t, C.c_uint]
                lib.cudaFreeHost.restype = C.c_int32
                lib.cudaFreeHost.argtypes = [C.c_void_p]
                self.lib = lib
                return
            except OSError:
                continue

    def alloc(self, nbytes: int):
        if self.lib is None:
            return None
        p = C.c_void_p()
        if self.lib.cudaHostAlloc(C.byref(p), C.c_size_t(nbytes), C.c_uint(0)) != 0:
            return None
        return p

    def free(self, p) -> None:
        if self.lib is not None and p:
            self.lib.cudaFreeHost(p)


def load_dll(path: Path):
    if not path.exists():
        raise SystemExit(f"未找到 {path}，请先运行 Algorithms\\undistortion\\scripts\\build_native.ps1")
    prepare_dll_dirs()
    d = C.CDLL(str(path))
    d.undist_create.restype = C.c_int32
    d.undist_create.argtypes = [C.c_char_p, C.c_int32, C.c_int32,
                                C.POINTER(UndistOptions), C.c_char_p, C.POINTER(C.c_void_p)]
    d.undist_process.restype = C.c_int32
    d.undist_process.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_void_p, C.c_int32,
                                 C.c_int32, C.POINTER(UndistTiming)]
    d.undist_get_info.restype = C.c_int32
    d.undist_get_info.argtypes = [C.c_void_p, C.POINTER(UndistInfo)]
    d.undist_get_error.restype = C.c_char_p
    d.undist_get_error.argtypes = [C.c_void_p]
    d.undist_release.restype = None
    d.undist_release.argtypes = [C.c_void_p]
    return d


# --------------------------------------------------------------------- image IO
def imread_gray(path: Path) -> np.ndarray | None:
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def imwrite_bmp(path: Path, img: np.ndarray) -> int:
    ok, buf = cv2.imencode(".bmp", img)
    if not ok:
        raise RuntimeError(f"imencode 失败: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return len(buf)


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "dataset/davisson/cpu16/pic_cpo16/ee2-pic/raw")
    ap.add_argument("--out", type=Path, default=ROOT / "dataset/davisson/cpu16/pic_cpo16/ee2-pic/undis")
    ap.add_argument("--matrix", type=Path, default=MATRIX_DIR)
    ap.add_argument("--width", type=int, default=5120)
    ap.add_argument("--height", type=int, default=4096)
    ap.add_argument("--pattern", default="*.bmp")
    ap.add_argument("--cache", type=Path, default=CACHE / "undist_5120x4096.engine",
                    help="引擎缓存路径（空字符串=不缓存）")
    ap.add_argument("--no-pinned", action="store_true", help="禁用 cudaHostAlloc，走 pageable 对照")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selfcheck", action="store_true",
                    help="用 recipe/raw/pic 去畸变后与 recipe/test/pic 对比，验证矩阵与调用正确")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    dll = load_dll(DLL)
    print(f"[dll] {DLL}  sizeof options/info/timing = "
          f"{C.sizeof(UndistOptions)}/{C.sizeof(UndistInfo)}/{C.sizeof(UndistTiming)}", flush=True)

    opt = UndistOptions(memory_mode=1, device_id=0, channels=1, log_level=1)
    if args.no_pinned:
        opt.memory_mode = 0
    cache = str(args.cache) if args.cache else None
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)

    ctx = C.c_void_p()
    t0 = time.perf_counter()
    rc = dll.undist_create(str(args.matrix).encode("utf-8"), args.width, args.height,
                           C.byref(opt), cache.encode("utf-8") if cache else None, C.byref(ctx))
    init_ms = (time.perf_counter() - t0) * 1000.0
    if rc != UNDIST_OK:
        err = dll.undist_get_error(ctx)
        if ctx:
            dll.undist_release(ctx)
        raise SystemExit(f"undist_create rc={rc}: {err.decode('utf-8', 'replace')}")
    info = UndistInfo()
    dll.undist_get_info(ctx, C.byref(info))
    print(f"[cfg] init {init_ms:.0f} ms (table {info.init_table_ms:.0f} + engine "
          f"{info.init_engine_ms:.0f}) | {args.width}x{args.height} | table "
          f"{info.table_bytes/2**20:.0f} MiB ({info.table_bytes_per_pixel} B/px) | device "
          f"{info.device_bytes/2**20:.0f} MiB | {info.gpu_name.decode('utf-8','replace')} "
          f"SM={info.sm_count} TRT={info.tensorrt_version.decode('utf-8','replace')}", flush=True)

    # 页锁缓冲（一次分配，反复复用）
    npx = args.width * args.height
    cudart = Cudart() if not args.no_pinned else Cudart.__new__(Cudart)
    if args.no_pinned or cudart.lib is None:
        cudart.lib = None
    src_p = cudart.alloc(npx) if cudart.lib is not None else None
    dst_p = cudart.alloc(npx) if cudart.lib is not None else None
    pinned = src_p is not None and dst_p is not None
    if pinned:
        print("[mem] cudaHostAlloc 2 x %.1f MiB → host_flags=1（零拷贝）" % (npx / 2**20), flush=True)
    else:
        src_host = np.zeros(npx, np.uint8)
        dst_host = np.zeros(npx, np.uint8)
        src_p = C.cast(src_host.ctypes.data_as(C.c_void_p), C.c_void_p)
        dst_p = C.cast(dst_host.ctypes.data_as(C.c_void_p), C.c_void_p)
        print("[mem] pageable（host_flags=0）", flush=True)

    timing = UndistTiming()

    def undistort(img: np.ndarray) -> np.ndarray:
        if pinned:
            C.memmove(src_p, img.ctypes.data, npx)
            rc2 = dll.undist_process(ctx, src_p, args.width, dst_p, args.width, 1, C.byref(timing))
            if rc2 != UNDIST_OK:
                raise RuntimeError(f"undist_process rc={rc2}: "
                                   f"{dll.undist_get_error(ctx).decode('utf-8','replace')}")
            return np.frombuffer((C.c_ubyte * npx).from_address(dst_p.value),
                                 np.uint8).reshape(args.height, args.width).copy()
        src_host[:] = img.reshape(-1)
        rc2 = dll.undist_process(ctx, src_p, args.width, dst_p, args.width, 0, C.byref(timing))
        if rc2 != UNDIST_OK:
            raise RuntimeError(f"undist_process rc={rc2}: "
                               f"{dll.undist_get_error(ctx).decode('utf-8','replace')}")
        return dst_host.reshape(args.height, args.width).copy()

    try:
        # ---------------- 自检：recipe/raw/pic → 对比 recipe/test/pic ----------------
        if args.selfcheck:
            raw = ROOT / "dataset/davisson/cpu16/pic_cpo16/recipe/raw/pic"
            ref = ROOT / "dataset/davisson/cpu16/pic_cpo16/recipe/test/pic"
            print("\n=== 自检：raw/pic 去畸变 vs test/pic（应几乎一致）===", flush=True)
            for name in ("PIC_1_1.bmp", "PIC_2_5.bmp", "PIC_3_10.bmp"):
                p, q = raw / name, ref / name
                if not (p.exists() and q.exists()):
                    print(f"  {name}: 缺文件，跳过")
                    continue
                im, rf = imread_gray(p), imread_gray(q)
                out = undistort(im)
                d = np.abs(out.astype(np.int16) - rf.astype(np.int16))
                print(f"  {name}: exact={float((d==0).mean())*100:6.2f}% "
                      f"mean|d|={float(d.mean()):.4f} max={int(d.max())} "
                      f"({timing.total_ms:.1f} ms: h2d {timing.h2d_ms:.1f} infer {timing.infer_ms:.1f} "
                      f"d2h {timing.d2h_ms:.1f})", flush=True)

        # ---------------- 批量 ----------------
        cases = sorted([d for d in args.raw.iterdir() if d.is_dir()])
        if not cases:
            raise SystemExit(f"{args.raw} 下没有子目录")
        report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "raw": str(args.raw), "out": str(args.out), "matrix": str(args.matrix),
                  "pinned": pinned, "size": [args.width, args.height],
                  "init_ms": round(init_ms, 1),
                  "init_breakdown_ms": {"table": round(info.init_table_ms, 1),
                                        "engine": round(info.init_engine_ms, 1)},
                  "table_bytes": int(info.table_bytes), "device_bytes": int(info.device_bytes),
                  "cases": []}
        t_all = time.perf_counter()
        total_imgs = 0
        totals: list[float] = []
        for ci, case_dir in enumerate(cases, 1):
            files = sorted(case_dir.glob(args.pattern))
            if args.limit:
                files = files[:args.limit]
            out_dir = args.out / case_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[{ci}/{len(cases)}] {case_dir.name}: {len(files)} 张", flush=True)
            per_ms: list[float] = []
            h2d: list[float] = []
            inf: list[float] = []
            d2h: list[float] = []
            t_case = time.perf_counter()
            for k, p in enumerate(files, 1):
                im = imread_gray(p)
                if im is None:
                    print(f"  !! 读图失败 {p.name}")
                    continue
                if im.ndim != 2:
                    raise SystemExit(f"{p} 不是单通道（shape={im.shape}），去畸变库只支持 8UC1")
                if im.shape != (args.height, args.width):
                    raise SystemExit(f"{p.name} 尺寸 {im.shape} != {(args.height, args.width)}")
                out = undistort(im)
                imwrite_bmp(out_dir / p.name, out)
                per_ms.append(timing.total_ms)
                h2d.append(timing.h2d_ms)
                inf.append(timing.infer_ms)
                d2h.append(timing.d2h_ms)
                if k % 10 == 0 or k == len(files):
                    print(f"  {k}/{len(files)}  mean total {statistics.mean(per_ms):.2f} ms "
                          f"(h2d {statistics.mean(h2d):.2f} infer {statistics.mean(inf):.2f} "
                          f"d2h {statistics.mean(d2h):.2f})", flush=True)
            case_ms = (time.perf_counter() - t_case) * 1000.0
            total_imgs += len(per_ms)
            totals += per_ms
            report["cases"].append({
                "case": case_dir.name, "images": len(per_ms), "wall_ms": round(case_ms, 1),
                "per_image_total_ms": round(statistics.mean(per_ms), 3) if per_ms else 0,
                "per_image_h2d_ms": round(statistics.mean(h2d), 3) if h2d else 0,
                "per_image_infer_ms": round(statistics.mean(inf), 3) if inf else 0,
                "per_image_d2h_ms": round(statistics.mean(d2h), 3) if d2h else 0,
                "out_dir": str(out_dir),
            })
        report["images"] = total_imgs
        report["elapsed_s"] = round(time.perf_counter() - t_all, 1)
        if totals:
            report["per_image_total_ms_median"] = round(statistics.median(totals), 3)
        rp = args.report or (args.out / "undist_report.json")
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n=== 汇总 ===")
        print(f"{'case':8s} {'images':>6s} {'wall_s':>7s} {'h2d':>6s} {'infer':>6s} {'d2h':>6s} {'total':>7s}")
        for c in report["cases"]:
            print(f"{c['case']:8s} {c['images']:6d} {c['wall_ms']/1000:7.1f} "
                  f"{c['per_image_h2d_ms']:6.2f} {c['per_image_infer_ms']:6.2f} "
                  f"{c['per_image_d2h_ms']:6.2f} {c['per_image_total_ms']:7.2f}")
        print(f"\n共 {total_imgs} 张，耗时 {report['elapsed_s']} s，单张中位 "
              f"{report.get('per_image_total_ms_median', 0)} ms | 报告 {rp}")
        return 0
    finally:
        dll.undist_release(ctx)
        if pinned:
            cudart.free(src_p)
            cudart.free(dst_p)


if __name__ == "__main__":
    raise SystemExit(main())
