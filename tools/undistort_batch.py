# -*- coding: utf-8 -*-
"""
undistort_batch.py —— 批量去畸变工具（走生产 DLL 的 C 接口）

特点
  * 直接调用 CAPP/build/bin/Undistortion.dll 导出的
    AOI_Undistortion_Create/SetConfig/Infer/Destroy，
    与 C# 侧 P/Invoke 完全同一条代码路径，不做任何算法二次实现。
  * 全 Unicode 路径安全：读用 np.fromfile + cv2.imdecode，
    写用 cv2.imencode + ndarray.tofile（cv2.imread/imwrite 在含中文路径上会失败）。
  * 输出与原图同目录，文件名插入后缀（默认 _undis）。
  * 幂等：已存在的 *_undis.bmp 默认跳过（--overwrite 可强制重做）。

用法
  python tools/undistort_batch.py                       # 处理 1..9 子目录
  python tools/undistort_batch.py --dirs 1,3,5          # 只处理指定子目录
  python tools/undistort_batch.py --suffix _undis2      # 换后缀
  python tools/undistort_batch.py --overwrite           # 覆盖已有结果
  python tools/undistort_batch.py --matrix <另一个矩阵目录>
"""
import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np
import cv2

PROJ = r"E:\algrithm\project\botong\davissionNew\Algorithms\undistortion_standaloneV0"
DLL_PATH = os.path.join(PROJ, "CAPP", "build", "bin", "Undistortion.dll")
OPENCV_BIN = r"E:\3rdparty\opencv\build\x64\vc16\bin"
DEFAULT_MATRIX = os.path.join(PROJ, "botong25")
DEFAULT_ROOT = (r"E:\algrithm\project\botong\davissionNew\Algorithms"
                r"\deployV1\pic\pic_demo-ee\wg边缘缺陷")

MATRIX_W, MATRIX_H = 5120, 4096


# ---------------------------------------------------------------- C ABI
class AoiUndistortionConfig(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("matrixPath", ctypes.c_char * 512),
                ("matrixWidth", ctypes.c_int),
                ("matrixHeight", ctypes.c_int),
                ("outputChannels", ctypes.c_int)]


class AoiBaseResult(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("errorCode", ctypes.c_int),
                ("errorMessage", ctypes.c_char * 256),
                ("processingTime", ctypes.c_float)]


def load_dll():
    if not os.path.isfile(DLL_PATH):
        raise FileNotFoundError(
            f"DLL 不存在: {DLL_PATH}\n请先执行 tools\\build_release.bat")
    if os.path.isdir(OPENCV_BIN):
        try:
            os.add_dll_directory(OPENCV_BIN)
        except Exception:
            pass
    dll = ctypes.CDLL(DLL_PATH)
    dll.AOI_Undistortion_Create.restype = ctypes.c_void_p
    dll.AOI_Undistortion_Create.argtypes = []
    dll.AOI_Undistortion_Destroy.restype = None
    dll.AOI_Undistortion_Destroy.argtypes = [ctypes.c_void_p]
    dll.AOI_Undistortion_SetConfig.restype = ctypes.c_int
    dll.AOI_Undistortion_SetConfig.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(AoiUndistortionConfig)]
    dll.AOI_Undistortion_IsConfigured.restype = ctypes.c_int
    dll.AOI_Undistortion_IsConfigured.argtypes = [ctypes.c_void_p]
    dll.AOI_Undistortion_Infer.restype = ctypes.c_int
    dll.AOI_Undistortion_Infer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ctypes.POINTER(AoiBaseResult)]
    return dll


class Undistorter:
    def __init__(self, matrix_dir, log=print):
        self.dll = load_dll()
        self.h = self.dll.AOI_Undistortion_Create()
        if not self.h:
            raise RuntimeError("AOI_Undistortion_Create 失败")
        cfg = AoiUndistortionConfig()
        raw = os.fsencode(matrix_dir)[:511]          # 用 ANSI/本地编码传路径，与 C++ 侧一致
        cfg.matrixPath = raw
        cfg.matrixWidth = MATRIX_W
        cfg.matrixHeight = MATRIX_H
        cfg.outputChannels = 1
        t0 = time.perf_counter()
        rc = self.dll.AOI_Undistortion_SetConfig(self.h, ctypes.byref(cfg))
        self.init_ms = (time.perf_counter() - t0) * 1000.0
        if rc != 0:
            self.close()
            raise RuntimeError(f"AOI_Undistortion_SetConfig 失败, rc={rc}")
        if self.dll.AOI_Undistortion_IsConfigured(self.h) == 0:
            self.close()
            raise RuntimeError("AOI_Undistortion_IsConfigured == 0")
        log(f"  [init] matrix = {matrix_dir}")
        log(f"  [init] setConfig = {self.init_ms:.0f} ms")

    def process(self, gray):
        h, w = gray.shape
        if (w, h) != (MATRIX_W, MATRIX_H):
            raise ValueError(f"尺寸必须为 {MATRIX_W}x{MATRIX_H}，实际 {w}x{h}")
        out = np.empty((h, w), np.uint8)
        res = AoiBaseResult()
        rc = self.dll.AOI_Undistortion_Infer(
            self.h,
            gray.ctypes.data, w, h, out.ctypes.data,
            ctypes.byref(res))
        if rc != 0:
            msg = res.errorMessage.decode("utf-8", "replace")
            raise RuntimeError(f"AOI_Undistortion_Infer 失败 rc={rc}: {msg}")
        return out, float(res.processingTime)

    def close(self):
        if getattr(self, "h", None):
            self.dll.AOI_Undistortion_Destroy(self.h)
            self.h = None


# ---------------------------------------------------------------- 文件
def imread_u8(path):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def imwrite_u8(path, gray):
    ok, enc = cv2.imencode(".bmp", gray)
    if not ok:
        raise RuntimeError(f"imencode 失败: {path}")
    enc.tofile(path)


def pick_dirs(root, only):
    subs = [d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.isdigit()]
    subs.sort(key=int)
    if not subs:
        return ["."]     # 平铺目录（如 pic_demo-oe）：把根目录本身当成唯一一组
    if only:
        want = {s.strip() for s in only.split(",") if s.strip()}
        subs = [d for d in subs if d in want]
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--matrix", default=DEFAULT_MATRIX)
    ap.add_argument("--suffix", default="_undis")
    ap.add_argument("--dirs", default="", help="例如 1,2,3；默认全部数字子目录")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"根目录不存在: {args.root}")

    dirs = pick_dirs(args.root, args.dirs)
    if not dirs:
        sys.exit("没有找到 1..N 数字子目录")

    print("=" * 78)
    print(f"根目录 : {args.root}")
    print(f"子目录 : {dirs}")
    print(f"矩阵   : {args.matrix}")
    print(f"后缀   : {args.suffix}")
    print(f"DLL    : {DLL_PATH}")
    print("=" * 78)

    ud = None
    rows = []
    t_all0 = time.perf_counter()
    try:
        for d in dirs:
            dpath = args.root if d == "." else os.path.join(args.root, d)
            files = sorted(f for f in os.listdir(dpath)
                           if f.lower().endswith(".bmp")
                           and os.path.isfile(os.path.join(dpath, f))
                           and not os.path.splitext(f)[0].endswith(args.suffix))
            if not files:
                print(f"\n[{d}] 无待处理 bmp，跳过")
                continue

            if ud is None:
                print("\n[初始化去畸变器]")
                ud = Undistorter(args.matrix)

            print(f"\n[{d}] {len(files)} 张")
            for f in files:
                src = os.path.join(dpath, f)
                stem, ext = os.path.splitext(f)
                dst = os.path.join(dpath, f"{stem}{args.suffix}{ext}")
                if os.path.exists(dst) and not args.overwrite:
                    print(f"  - {f} -> 已存在，跳过（--overwrite 可覆盖）")
                    continue

                t0 = time.perf_counter()
                img = imread_u8(src)
                if img is None:
                    print(f"  ! {f} 读取失败，跳过")
                    rows.append({"dir": d, "file": f, "status": "READ_FAIL"})
                    continue
                if img.shape != (MATRIX_H, MATRIX_W):
                    print(f"  ! {f} 尺寸 {img.shape[::-1]} != ({MATRIX_W},{MATRIX_H})，跳过")
                    rows.append({"dir": d, "file": f, "status": "SIZE_MISMATCH",
                                 "shape": list(img.shape)})
                    continue

                out, lib_ms = ud.process(img)
                imwrite_u8(dst, out)
                total_ms = (time.perf_counter() - t0) * 1000.0
                src_b = os.path.getsize(src)
                dst_b = os.path.getsize(dst)
                print(f"  + {f} -> {os.path.basename(dst)}  "
                      f"infer={lib_ms:7.2f} ms  total={total_ms:7.2f} ms  "
                      f"{src_b:,} B -> {dst_b:,} B")
                rows.append({"dir": d, "file": f, "out": os.path.basename(dst),
                             "status": "OK", "infer_ms": lib_ms, "total_ms": total_ms,
                             "src_bytes": src_b, "dst_bytes": dst_b})
    finally:
        if ud is not None:
            ud.close()

    t_all = (time.perf_counter() - t_all0) * 1000.0
    ok = [r for r in rows if r.get("status") == "OK"]
    print("\n" + "=" * 78)
    print(f"完成：{len(ok)} / {len(rows)} 张成功，总耗时 {t_all/1000:.2f} s")
    if ok:
        inf = [r["infer_ms"] for r in ok]
        print(f"       infer 均值 {sum(inf)/len(inf):.2f} ms，"
              f"min {min(inf):.2f}，max {max(inf):.2f}")
    bad = [r for r in rows if r.get("status") != "OK"]
    if bad:
        print(f"       ⚠ 失败 {len(bad)} 项：")
        for r in bad:
            print(f"         {r['dir']}\\{r['file']}  {r['status']}")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump({"root": args.root, "matrix": args.matrix,
                       "suffix": args.suffix, "dirs": dirs,
                       "total_ms": t_all, "items": rows},
                      fp, indent=2, ensure_ascii=False)
        print("->", args.json)

    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
