# -*- coding: utf-8 -*-
"""
mask_frame_check.py —— 判定 mask 是在「原图(带畸变)」还是「去畸变图」坐标系下绘制的

可用素材（已用哈希确认）：
  raw/pic/*.bmp            == 原图以及mask/定拍取图/A8/*.bmp   40 张，逐字节相同，无黑边 -> 真正的 raw
  原图以及mask/A2ELongPI/*.bmp  39 张，带 10/11/19/20 黑边       -> 去畸变图
  raw/mask/*               == 原图以及mask/A2ELongPIMasks/*    193/193 字节相同 -> 同一套 mask

判别原理：
  mask 的边界是沿图像特征（区域分界/走线/开窗）勾出来的。
  因此在「mask 绘制时所用的那一帧」里，mask 边界应当紧贴图像强边缘；
  换到另一帧（两者相差最多 31 px 的平滑畸变位移），边界会偏离。
  指标：
    ① Chamfer 距离 —— mask 边界像素到图像边缘图的平均最近距离（越小越贴合）
    ② 边界处梯度均值 —— 在 mask 边界上取 |∇I| 的均值（越大越贴合）
  两种指标互相印证。
"""
import json
import os
import numpy as np
import cv2

B = r"E:\algrithm\project\botong\davissionNew\dataset\davisson\cpu16\pic_cpo16\recipe"
RAW_PIC = os.path.join(B, "raw", "pic")
UND_PIC = os.path.join(B, "原图以及mask", "A2ELongPI")
MASK = os.path.join(B, "原图以及mask", "A2ELongPIMasks")
OUT = (r"E:\algrithm\project\botong\davissionNew\Algorithms"
       r"\undistortion_standaloneV0\document\data\mask_frame_check.json")


def rd(p, flag=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def edge_map(img, thr_pct=92):
    """Sobel 幅值 + 百分位阈值 -> 二值边缘图"""
    g = cv2.GaussianBlur(img, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    thr = np.percentile(mag, thr_pct)
    return (mag > thr).astype(np.uint8), mag


def mask_boundary(m):
    """二值 mask 的边界带（单像素级）"""
    b = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return b > 0


def score(mask, img, edge_bin, grad_mag):
    b = mask_boundary(mask)
    n = int(b.sum())
    if n < 2000:                      # 边界太短，指标不可信
        return None
    # ① Chamfer：边界像素到最近图像边缘的平均距离
    dist = cv2.distanceTransform(1 - edge_bin, cv2.DIST_L2, 5)
    chamfer = float(dist[b].mean())
    # ② 边界处梯度均值（归一化到 0..1，便于跨图对比）
    gmean = float(grad_mag[b].mean() / (grad_mag.max() + 1e-9))
    return {"boundary_px": n, "chamfer": chamfer, "grad_norm": gmean}


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    stems = sorted(d for d in os.listdir(MASK)
                   if d.startswith("PIC_") and os.path.isdir(os.path.join(MASK, d)))

    report = {"per_mask": [], "summary": {}}
    tally = {"chamfer_better": {"raw": 0, "undis": 0, "tie": 0},
             "grad_better": {"raw": 0, "undis": 0, "tie": 0}}

    print(f"{'image':10s} {'mask':20s} {'Chamfer raw':>12s} {'Chamfer undis':>14s} "
          f"{'Grad raw':>9s} {'Grad undis':>11s}  verdict")
    print("-" * 104)

    for stem in stems:
        pr = os.path.join(RAW_PIC, stem + ".bmp")
        pu = os.path.join(UND_PIC, stem + ".bmp")
        if not (os.path.isfile(pr) and os.path.isfile(pu)):
            continue
        ir = rd(pr)
        iu = rd(pu)
        er, gr = edge_map(ir)
        eu, gu = edge_map(iu)

        for f in sorted(os.listdir(os.path.join(MASK, stem))):
            if not f.lower().endswith((".png", ".jpg")):
                continue
            mp = os.path.join(MASK, stem, f)
            m = rd(mp)
            if m is None or m.shape != ir.shape:
                continue
            mb = (m > 127).astype(np.uint8)
            if mb.sum() == 0:                 # 全零占位文件（如 whole_mask.png）
                continue
            sr = score(mb, ir, er, gr)
            su = score(mb, iu, eu, gu)
            if sr is None or su is None:
                continue

            # Chamfer 越小越贴合；Grad 越大越贴合
            if abs(sr["chamfer"] - su["chamfer"]) < 1e-3:
                cv = "tie"
            else:
                cv = "raw" if sr["chamfer"] < su["chamfer"] else "undis"
            tally["chamfer_better"][cv] += 1

            if abs(sr["grad_norm"] - su["grad_norm"]) < 1e-4:
                gv = "tie"
            else:
                gv = "raw" if sr["grad_norm"] > su["grad_norm"] else "undis"
            tally["grad_better"][gv] += 1

            print(f"{stem:10s} {f:20s} {sr['chamfer']:12.4f} {su['chamfer']:14.4f} "
                  f"{sr['grad_norm']:9.5f} {su['grad_norm']:11.5f}  chamfer:{cv:5s} grad:{gv}")
            report["per_mask"].append({
                "image": stem, "mask": f,
                "raw": sr, "undis": su,
                "chamfer_verdict": cv, "grad_verdict": gv})

    # 汇总：用像素数加权
    tot = len(report["per_mask"])
    cr = np.mean([x["raw"]["chamfer"] for x in report["per_mask"]])
    cu = np.mean([x["undis"]["chamfer"] for x in report["per_mask"]])
    gr = np.mean([x["raw"]["grad_norm"] for x in report["per_mask"]])
    gu = np.mean([x["undis"]["grad_norm"] for x in report["per_mask"]])
    report["summary"] = {
        "masks_compared": tot,
        "mean_chamfer_raw": float(cr), "mean_chamfer_undis": float(cu),
        "mean_grad_raw": float(gr), "mean_grad_undis": float(gu),
        "chamfer_votes": tally["chamfer_better"],
        "grad_votes": tally["grad_better"],
    }
    print("-" * 104)
    print(f"参与比较 mask 数: {tot}")
    print(f"Chamfer 均值: raw={cr:.4f}  undis={cu:.4f}  "
          f"-> {'去畸变' if cu < cr else '原图'} 更贴合 ({(cr-cu)/cr*100:+.1f}%)")
    print(f"边界梯度均值: raw={gr:.5f}  undis={gu:.5f}  "
          f"-> {'去畸变' if gu > gr else '原图'} 更贴合 ({(gu-gr)/max(gr,1e-9)*100:+.1f}%)")
    print(f"逐 mask 投票 Chamfer: {tally['chamfer_better']}")
    print(f"逐 mask 投票 Grad   : {tally['grad_better']}")

    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
