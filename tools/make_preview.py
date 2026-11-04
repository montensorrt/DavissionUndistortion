# -*- coding: utf-8 -*-
"""make_preview.py —— 生成 raw vs output0 的对比缩略图，用于人工确认几何正确性。"""
import os
import cv2
import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(PROJ, "demo", "raw")
OUT = os.path.join(PROJ, "demo", "output0")
DST = os.path.join(PROJ, "document", "data", "preview")
os.makedirs(DST, exist_ok=True)

NAMES = ["PIC_1_1.bmp", "PIC_2_5.bmp", "STEP_1_1.bmp"]
SCALE = 0.22


def load(p):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    assert img is not None, p
    return img


rows = []
for n in NAMES:
    a = load(os.path.join(RAW, n))
    b = load(os.path.join(OUT, n))
    ha, wa = a.shape
    a2 = cv2.resize(a, (int(wa * SCALE), int(ha * SCALE)), interpolation=cv2.INTER_AREA)
    b2 = cv2.resize(b, (int(wa * SCALE), int(ha * SCALE)), interpolation=cv2.INTER_AREA)
    cv2.putText(a2, "raw " + n, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    cv2.putText(b2, "undistorted", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2, cv2.LINE_AA)
    pair = np.hstack([a2, np.full((a2.shape[0], 8), 255, np.uint8), b2])
    rows.append(pair)

wmax = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1])), constant_values=255) for r in rows]
sheet = np.vstack([np.vstack([r, np.full((8, wmax), 255, np.uint8)]) for r in rows])
out = os.path.join(DST, "raw_vs_undistorted.png")
cv2.imwrite(out, sheet)
print("wrote", out, sheet.shape)

# 局部 1:1 裁剪对比（看锐度/边缘）
a = load(os.path.join(RAW, NAMES[0]))
b = load(os.path.join(OUT, NAMES[0]))
y0, x0, s = 1600, 2000, 700
ca = a[y0:y0 + s, x0:x0 + s]
cb = b[y0:y0 + s, x0:x0 + s]
crop = np.hstack([ca, np.full((s, 8), 255, np.uint8), cb])
out2 = os.path.join(DST, "crop_1to1.png")
cv2.imwrite(out2, crop)
print("wrote", out2, crop.shape)
