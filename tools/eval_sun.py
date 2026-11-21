"""Napos stack kiértékelése a szintetikus „igazsághoz” képest.

Használat: python tools/eval_sun.py <stack.tif> <truth_mappa> [--out ábra.png]
Mér:
  - részlethűség: a DoG(σ1–σ6) részletsáv korrelációja az igazsággal a korongon belül
  - napfoltok: az umbra mélysége (1 - min/környezet) az eredményben és az igazságban, foltonként
  - hamis foltok: sötét pöttyök (≥ 4% mély, 2–12 px) az eredményben ott, ahol az igazságban nincs folt
"""
import sys
import json
import argparse

import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("stack")
ap.add_argument("truth")
ap.add_argument("--out")
a = ap.parse_args()

st = cv2.imread(a.stack, cv2.IMREAD_UNCHANGED).astype(np.float32)
st = st / (65535.0 if st.max() > 255 else 255.0)
g = cv2.cvtColor(st, cv2.COLOR_BGR2GRAY) if st.ndim == 3 else st
gt = cv2.imread(a.truth + "/ground_truth.png", 0).astype(np.float32) / 255.0
truth = json.load(open(a.truth + "/truth.json", encoding="utf-8"))


def disc(img):
    m = (img > 0.3 * np.percentile(img, 99)).astype(np.uint8)
    n, lab, st_, ce = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st_[1:, cv2.CC_STAT_AREA]))
    m = lab == k
    return m, ce[k], np.sqrt(m.sum() / np.pi)


mg, cg, rg = disc(g)
mt, ct, rt = disc(gt)
# kezdeti becslés: középpont és sugár → affin, majd ECC finomítás (GT -> eredmény)
s = rg / rt
A0 = np.array([[s, 0, cg[0] - s * ct[0]], [0, s, cg[1] - s * ct[1]]], np.float32)
gn = g / np.percentile(g[mg], 95); tn = gt / np.percentile(gt[mt], 95)
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
warp = A0.copy()
for sc in (0.25, 0.5, 1.0):
    gs = cv2.resize(gn, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    tw = cv2.warpAffine(tn, warp, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)
    ts = cv2.resize(tw, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    dW = np.eye(2, 3, dtype=np.float32)
    try:
        _, dW = cv2.findTransformECC(cv2.GaussianBlur(ts, (0, 0), 1.5), cv2.GaussianBlur(gs, (0, 0), 1.5), dW,
                                     cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        pass
    # dW: ts-koordináták -> gs-koordináták (skálázva) ; teljes felbontásra:
    dW[:, 2] /= sc
    M = np.vstack([warp, [0, 0, 1]]); D = np.vstack([dW, [0, 0, 1]])
    warp = (D @ M)[:2].astype(np.float32)
tw = cv2.warpAffine(tn, warp, (g.shape[1], g.shape[0]), flags=cv2.INTER_CUBIC)
inner = cv2.erode(mg.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)


def dog(x):
    return cv2.GaussianBlur(x, (0, 0), 1.0) - cv2.GaussianBlur(x, (0, 0), 6.0)


dg, dt = dog(gn), dog(tw)
corr = float(np.corrcoef(dg[inner], dt[inner])[0, 1])

# napfoltok: helyük az eredményben
res_spots = []
for sp in truth["spots"]:
    p = warp @ np.array([sp["x"], sp["y"], 1.0])
    x, y = int(round(p[0])), int(round(p[1]))
    ru = sp["r_umbra"] * s
    w = int(max(8, 4 * sp["r_penumbra"] * s))
    if not (w <= x < g.shape[1] - w and w <= y < g.shape[0] - w):
        continue

    def depth(img):
        box = img[y - w:y + w, x - w:x + w]
        ring = np.percentile(box, 75)
        core = img[y - int(ru) - 1:y + int(ru) + 2, x - int(ru) - 1:x + int(ru) + 2].min()
        return 1 - core / max(ring, 1e-6)
    res_spots.append(dict(r_umbra=sp["r_umbra"], depth_stack=float(depth(gn)), depth_truth=float(depth(tw))))

# hamis foltok: sötét pöttyök az eredményben, amelyek az igazságban nincsenek
bg = cv2.GaussianBlur(gn, (0, 0), 8)
rel = gn / np.maximum(bg, 1e-3)
relt = tw / np.maximum(cv2.GaussianBlur(tw, (0, 0), 8), 1e-3)
cand = ((rel < 0.96) & inner).astype(np.uint8)
n, lab, sts, ces = cv2.connectedComponentsWithStats(cand)
false_spots = []
for j in range(1, n):
    area = sts[j, cv2.CC_STAT_AREA]
    if area < 4 or area > 600:
        continue
    cx_, cy_ = ces[j]
    yy0, xx0 = int(cy_), int(cx_)
    win = relt[max(0, yy0 - 10):yy0 + 11, max(0, xx0 - 10):xx0 + 11]
    if win.size and win.min() < 0.97:
        continue  # az igazságban is van ott sötét folt
    false_spots.append(dict(x=float(cx_), y=float(cy_), area=int(area),
                            depth=float(1 - rel[lab == j].min())))

ratios = [r["depth_stack"] / max(r["depth_truth"], 1e-6) for r in res_spots]
out = dict(detail_corr=corr, spots=res_spots, spot_depth_ratio_median=float(np.median(ratios)) if ratios else None,
           spot_depth_ratio_min=float(np.min(ratios)) if ratios else None, false_spots=len(false_spots),
           false_spot_list=false_spots[:20])
print(json.dumps({k: v for k, v in out.items() if k not in ("spots", "false_spot_list")}, ensure_ascii=False))
for r in res_spots:
    print(f"  folt umbra {r['r_umbra']:>4} px: mélység eredmény {r['depth_stack']:.3f} / igazság {r['depth_truth']:.3f}")
for f in false_spots[:10]:
    print(f"  HAMIS folt: ({f['x']:.0f},{f['y']:.0f}) terület {f['area']} px, mélység {f['depth']:.3f}")
if a.out:
    vis = cv2.cvtColor(np.clip(gn * 200, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for f in false_spots:
        cv2.circle(vis, (int(f["x"]), int(f["y"])), 18, (0, 0, 255), 2)
    cv2.imwrite(a.out, vis)
