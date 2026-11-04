# -*- coding: utf-8 -*-
"""
make_testset.py —— 由 recipe 生成 test 集：多 mask 合并 + 图像去畸变

功能
  1) 把 raw/mask/<PIC_x_y>/ 下的全部 mask 合并为一张二值 mask，
     以目录名保存（PIC_1_1.png … PIC_3_10.png）→ <out-mask>
  2) 把 raw/pic/*.bmp 去畸变 → <out-pic>（同名 .bmp）
  3) 输出多 mask 合并的完整分析：灰度分布 / 阈值敏感性 / 覆盖率 /
     重叠层数直方图 / 逐对包含关系（判断"同一区域"还是"不同区域"）

关于合并语义（详见 --help 与报告）
  * 源 mask 并非严格二值：实测含 6~20 个灰度级（JPEG 压缩 + 多边形抗锯齿边），
    因此必须先按阈值二值化。默认 --thr 127，可用 --thr 调。
  * 同一像素可能被多个 mask 同时覆盖（实测最多 4 层），
    --mode union（默认）只表达"是否属于任一 ROI"，**丢失区域身份**；
    若下游需要区域语义，用 --mode label 输出标签图（每张 mask 一个 id）。

用法
  python tools/make_testset.py                       # 默认：并集二值 + 去畸变 raw/pic
  python tools/make_testset.py --thr 200             # 更严格的二值阈值
  python tools/make_testset.py --mode label          # 输出标签图（保留区域 id）
  python tools/make_testset.py --pic-source undis     # test/pic 直接用已去畸变的 A2ELongPI
"""
import argparse
import json
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from undistort_batch import Undistorter, imread_u8, imwrite_u8   # noqa: E402

PROJ = r"E:\algrithm\project\botong\davissionNew\Algorithms\undistortion_standaloneV0"
RECIPE = r"E:\algrithm\project\botong\davissionNew\dataset\davisson\cpu16\pic_cpo16\recipe"
DEFAULT_MATRIX = os.path.join(PROJ, "botong25")

W, H = 5120, 4096


def rd(path, flag=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(path, np.uint8), flag)


def wr(path, img):
    ok, enc = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise RuntimeError(f"imencode 失败: {path}")
    enc.tofile(path)


# ---------------------------------------------------------------- 分析工具
def gray_profile(mask):
    """灰度分布：返回 (唯一值数, 0/255 之外的中间值像素数, 主要取值)"""
    u, c = np.unique(mask, return_counts=True)
    mid = int(c[(u != 0) & (u != 255)].sum())
    top = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:4]
    return {"unique": int(u.size), "mid_pixels": mid,
            "values": [int(v) for v in u.tolist()][:24],
            "top_values": [int(v) for _, v in top]}


def threshold_sensitivity(masks, thrs=(1, 32, 64, 127, 160, 200)):
    """不同阈值下的并集覆盖率 —— 判断二值化是否稳定"""
    out = {}
    for t in thrs:
        un = np.zeros((H, W), bool)
        for m in masks:
            un |= (m > t)
        out[str(t)] = float(un.mean())
    return out


def pairwise_containment(masks, names, ds=8):
    """逐对包含关系（1/ds 降采样，足够分类）"""
    small = [(m[::ds, ::ds] > 127) for m in masks]
    res = []
    for i in range(len(small)):
        for j in range(i + 1, len(small)):
            a, b = small[i], small[j]
            inter = int((a & b).sum())
            if inter == 0:
                continue
            na, nb = int(a.sum()), int(b.sum())
            res.append({
                "a": names[i], "b": names[j],
                "inter": inter, "na": na, "nb": nb,
                "iou": inter / (na + nb - inter),
                "contain_min": inter / max(1, min(na, nb)),
                "contain_a_in_b": inter / max(1, na),
            })
    return res


def classify(pairs, nested_thr=0.90, partial_thr=0.05):
    nested, partial, cross = [], [], []
    for p in pairs:
        if p["contain_min"] >= nested_thr:
            nested.append(p)
        elif p["iou"] >= partial_thr:
            partial.append(p)
        else:
            cross.append(p)
    return nested, partial, cross


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default=RECIPE)
    ap.add_argument("--mask-root", default="")
    ap.add_argument("--pic-root", default="")
    ap.add_argument("--out-mask", default="")
    ap.add_argument("--out-pic", default="")
    ap.add_argument("--matrix", default=DEFAULT_MATRIX)
    ap.add_argument("--thr", type=int, default=127)
    ap.add_argument("--on-value", type=int, default=255)
    ap.add_argument("--mode", choices=["union", "label"], default="union")
    ap.add_argument("--pic-source", choices=["raw", "undis"], default="raw",
                    help="raw=对 raw/pic 去畸变；undis=直接用已去畸变的 A2ELongPI")
    ap.add_argument("--json", default="")
    ap.add_argument("--skip-analyze", action="store_true")
    args = ap.parse_args()

    mask_root = args.mask_root or os.path.join(args.recipe, "raw", "mask")
    pic_root = args.pic_root or os.path.join(args.recipe, "raw", "pic")
    out_mask = args.out_mask or os.path.join(args.recipe, "test", "mask")
    out_pic = args.out_pic or os.path.join(args.recipe, "test", "pic")
    undis_src = os.path.join(args.recipe, "原图以及mask", "A2ELongPI")

    if not os.path.isdir(mask_root):
        sys.exit(f"mask 根目录不存在: {mask_root}")

    os.makedirs(out_mask, exist_ok=True)
    os.makedirs(out_pic, exist_ok=True)

    stems = sorted((d for d in os.listdir(mask_root)
                    if d.startswith("PIC_") and os.path.isdir(os.path.join(mask_root, d))),
                   key=lambda s: (int(s.split("_")[1]), int(s.split("_")[2])))
    print("=" * 84)
    print(f"mask 源 : {mask_root}  ({len(stems)} 个子目录)")
    print(f"图像源 : {pic_root if args.pic_source=='raw' else undis_src}")
    print(f"输出    : mask={out_mask}   pic={out_pic}")
    print(f"阈值={args.thr}  模式={args.mode}  on_value={args.on_value}")
    print("=" * 84)

    report = {"thr": args.thr, "mode": args.mode, "on_value": args.on_value,
              "mask_root": mask_root, "pic_root": pic_root,
              "out_mask": out_mask, "out_pic": out_pic,
              "items": [], "analysis": {}}

    # ---------------- 1) 合并 mask ----------------
    print("\n[1/2] 合并 mask")
    layer_hist_all = np.zeros(8, np.int64)
    union_cov_all, sum_cov_all = [], []
    sens_sample = None
    pair_all = []

    for k, stem in enumerate(stems, 1):
        d = os.path.join(mask_root, stem)
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg")))
        if not files:
            print(f"  [{k:02d}/{len(stems)}] {stem} 无 mask，跳过")
            continue

        masks, names, empty = [], [], []
        for f in files:
            m = rd(os.path.join(d, f))
            if m is None or m.shape != (H, W):
                print(f"    ! {f} 尺寸异常，跳过")
                continue
            if int((m > args.thr).sum()) == 0:
                empty.append(f)          # 例如全零占位文件 whole_mask.png
            masks.append(m)
            names.append(f)

        if not masks:
            continue

        cnt = np.zeros((H, W), np.uint8)
        union = np.zeros((H, W), bool)
        for m in masks:
            b = m > args.thr
            union |= b
            cnt += b.astype(np.uint8)

        if args.mode == "union":
            out = (union.astype(np.uint8) * args.on_value)
        else:                              # label：按文件顺序编号，后写覆盖先写
            out = np.zeros((H, W), np.uint8)
            for idx, m in enumerate(masks, 1):
                out[(m > args.thr) & (out == 0)] = idx
        op = os.path.join(out_mask, stem + ".png")
        wr(op, out)

        # —— 统计 ——
        hist = np.bincount(cnt.ravel(), minlength=8)[:8]
        layer_hist_all += hist[:8]
        cov = float(union.mean())
        union_cov_all.append(cov)
        per = []
        for f, m in zip(names, masks):
            gp = gray_profile(m)
            per.append({"file": f, "coverage": float((m > args.thr).mean()),
                        "gray": gp})
            sum_cov_all.append(float((m > args.thr).mean()))
        pairs = pairwise_containment(masks, names) if not args.skip_analyze else []
        if pairs:
            nested, partial, cross = classify(pairs)
            pair_all.append({"stem": stem, "nested": len(nested), "partial": len(partial),
                             "cross": len(cross),
                             "nested_pairs": [(p["a"], p["b"], round(p["contain_min"], 3))
                                              for p in nested],
                             "partial_pairs": [(p["a"], p["b"], round(p["iou"], 3))
                                               for p in partial]})
        if sens_sample is None and len(masks) >= 8:
            sens_sample = {"stem": stem, "sens": threshold_sensitivity(masks)}

        print(f"  [{k:02d}/{len(stems)}] {stem}: {len(names)} 个 mask"
              f"{' (含 ' + str(len(empty)) + ' 个全零)' if empty else ''}  "
              f"并集覆盖={cov*100:.2f}%  "
              f"重叠 2层={int(hist[2]):,} 3+层={int(hist[2:].sum()):,}  max={int(hist.nonzero()[0].max()) if hist.any() else 0}")
        report["items"].append({
            "stem": stem, "n_masks": len(names), "empty_masks": empty,
            "union_coverage": cov,
            "coverage_sum": float(sum(x["coverage"] for x in per)),
            "layers": {"1": int(hist[1]), "2": int(hist[2]),
                       "3": int(hist[3]), "4+": int(hist[4:].sum())},
            "max_layers": int(hist.nonzero()[0].max()) if hist.any() else 0,
            "masks": per, "out": os.path.basename(op)})

    # ---------------- 2) 去畸变图像 ----------------
    print("\n[2/2] 生成 test/pic")
    if args.pic_source == "undis":
        import shutil
        n = 0
        for stem in stems:
            s = os.path.join(undis_src, stem + ".bmp")
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(out_pic, stem + ".bmp"))
                n += 1
        print(f"  直接拷贝已去畸变图 {n} 张 -> {out_pic}")
    else:
        ud = Undistorter(args.matrix)
        n = 0
        try:
            for k, stem in enumerate(stems, 1):
                s = os.path.join(pic_root, stem + ".bmp")
                if not os.path.isfile(s):
                    print(f"  [{k:02d}] {stem}.bmp 不存在，跳过")
                    continue
                img = imread_u8(s)
                if img is None or img.shape != (H, W):
                    print(f"  [{k:02d}] {stem} 尺寸异常，跳过")
                    continue
                out, lib_ms = ud.process(img)
                imwrite_u8(os.path.join(out_pic, stem + ".bmp"), out)
                n += 1
                print(f"  [{k:02d}/{len(stems)}] {stem}.bmp  infer={lib_ms:6.2f} ms")
        finally:
            ud.close()
        print(f"  去畸变 {n} 张 -> {out_pic}")

    # ---------------- 3) 汇总分析 ----------------
    tot = layer_hist_all[1:].sum()
    # 按目录口径对齐：每目录先求 Σ覆盖率，再对目录取均值（否则与并集均值不可比）
    cov_sum_per_dir = [it["coverage_sum"] for it in report["items"]]
    cov_union_per_dir = [it["union_coverage"] for it in report["items"]]
    uc = np.asarray(cov_union_per_dir) if cov_union_per_dir else np.asarray([0.0])
    sc = np.asarray(cov_sum_per_dir) if cov_sum_per_dir else np.asarray([0.0])
    analysis = {
        "mask_dirs": len(report["items"]),
        "union_coverage_mean": float(uc.mean()),
        "union_coverage_min": float(uc.min()),
        "union_coverage_median": float(np.median(uc)),
        "union_coverage_max": float(uc.max()),
        "dirs_union_over_90pct": int((uc > 0.90).sum()),
        "coverage_sum_mean": float(sc.mean()),
        "overlap_amplification": float(sc.mean() / uc.mean()) if uc.mean() > 0 else None,
        "pixels_covered_total": int(tot),
        "layer_share": {
            "1": float(layer_hist_all[1] / tot) if tot else 0.0,
            "2": float(layer_hist_all[2] / tot) if tot else 0.0,
            "3": float(layer_hist_all[3] / tot) if tot else 0.0,
            "4+": float(layer_hist_all[4:].sum() / tot) if tot else 0.0,
        },
        "threshold_sensitivity": sens_sample,
        "pairwise": pair_all,
    }
    report["analysis"] = analysis

    print("\n" + "=" * 84)
    print("多 mask 合并分析汇总")
    print(f"  mask 目录数             : {analysis['mask_dirs']}")
    print(f"  并集覆盖率 均值/最小/中位/最大 : {analysis['union_coverage_mean']*100:.2f}% / "
          f"{analysis['union_coverage_min']*100:.2f}% / {analysis['union_coverage_median']*100:.2f}% / "
          f"{analysis['union_coverage_max']*100:.2f}%")
    print(f"  并集覆盖率 > 90% 的目录  : {analysis['dirs_union_over_90pct']} / {analysis['mask_dirs']}")
    print(f"  Σ各mask覆盖率（按目录均值）: {analysis['coverage_sum_mean']*100:.2f}%")
    print(f"  重叠放大倍数 Σ/并集      : {analysis['overlap_amplification']:.3f}×"
          if analysis["overlap_amplification"] else "  重叠放大倍数             : n/a")
    ls = analysis["layer_share"]
    print(f"  被覆盖像素的层数占比     : 1层 {ls['1']*100:.1f}% / 2层 {ls['2']*100:.1f}% / "
          f"3层 {ls['3']*100:.1f}% / 4+层 {ls['4+']*100:.1f}%")
    if sens_sample:
        print(f"  阈值敏感性（{sens_sample['stem']}）:")
        for t, v in sens_sample["sens"].items():
            print(f"      thr>{t:<4s} 并集覆盖率 = {v*100:.3f}%")
    n_nested = sum(p["nested"] for p in pair_all)
    n_partial = sum(p["partial"] for p in pair_all)
    n_cross = sum(p["cross"] for p in pair_all)
    print(f"  mask 对关系              : 嵌套/同区域 {n_nested} 对, "
          f"部分相交 {n_partial} 对, 基本不相交 {n_cross} 对")
    print("=" * 84)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print("->", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
