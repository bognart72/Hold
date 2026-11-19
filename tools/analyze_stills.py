"""Önálló Hold-fotók diagnosztikája: hely, méret, fényesség, élesség, és geometria a legélesebbhez.

Az illesztés kétlépcsős: AKAZE jellemzőpontok + RANSAC (nagy eltolás/forgás is mehet),
majd ECC affin finomítás. Képkockánként homográfiát is becsül.

Használat: python analyze_stills.py <kep_mappa> <kimeneti_mappa>
"""
import sys, os, glob, csv
import numpy as np
import cv2
from PIL import Image, ImageOps

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
paths = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")) + glob.glob(os.path.join(src, "*.tif*")))
DS = 4

def load(p):
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # EXIF-forgatás alkalmazása
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

imgs, small, info = [], [], []
for p in paths:
    f = load(p); imgs.append(f)
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, None, fx=1/DS, fy=1/DS, interpolation=cv2.INTER_AREA)
    small.append(s)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[k]; H, W = s.shape
    moon = cv2.resize((lab == k).astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
    inner = cv2.erode(moon, np.ones((41, 41), np.uint8)).astype(bool)
    gb = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
    lap = cv2.Laplacian(gb, cv2.CV_32F)
    info.append(dict(file=os.path.basename(p), w=g.shape[1], h=g.shape[0], cx=ce[k][0]*DS, cy=ce[k][1]*DS,
                     bw=w*DS, bh=h*DS, touches=int(x <= 1 or y <= 1 or x+w >= W-1 or y+h >= H-1),
                     mean=float(g[inner].mean()), sat=float((g >= 250).mean()*100),
                     sharp=float(lap[inner].var() / gb[inner].mean()**2)))

best = int(np.argmax([r["sharp"] for r in info]))
akaze = cv2.AKAZE_create()
kb, db = akaze.detectAndCompute(small[best], None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
ref = cv2.GaussianBlur(small[best].astype(np.float32) / 255, (0, 0), 1.0)
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
for i, r in enumerate(info):
    k2, d2 = akaze.detectAndCompute(small[i], None)
    ms = [m for m, n in bf.knnMatch(db, d2, k=2) if m.distance < 0.8 * n.distance]
    P1 = np.float32([kb[m.queryIdx].pt for m in ms]); P2 = np.float32([k2[m.trainIdx].pt for m in ms])
    A, inl = cv2.estimateAffinePartial2D(P1, P2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    Hm, inlh = cv2.findHomography(P1, P2, cv2.RANSAC, 2.0)
    # ECC affin finomítás a jellemzőpontos becslésből indulva (ref -> kép)
    warp = A.astype(np.float32)
    try:
        cc, warp = cv2.findTransformECC(ref, cv2.GaussianBlur(small[i].astype(np.float32)/255, (0, 0), 1.0), warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        cc = np.nan
    M = warp[:, :2].astype(float)
    sx = np.hypot(M[0, 0], M[1, 0]); rot = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    sy = np.linalg.det(M) / sx; shear = (M[0, 0]*M[0, 1] + M[1, 0]*M[1, 1]) / sx**2
    # homográfia perspektív része: mekkora px-eltérést okoz a Hold szélén az affinhoz képest
    Hn = Hm / Hm[2, 2]
    persp = float(np.hypot(Hn[2, 0], Hn[2, 1]) * DS)  # 1/px a teljes felbontásra vetítve közelítőleg
    r.update(matches=len(ms), inliers=int(inl.sum()), ecc=cc, sx=sx, sy=sy, rot=rot, shear=shear,
             tx=warp[0, 2]*DS, ty=warp[1, 2]*DS, persp_px=persp * (r["bw"]/2)**2)

keys = list(info[0].keys())
with open(os.path.join(out, "stills.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(info)
for r in info:
    print(f"{r['file']}: c=({r['cx']:.0f},{r['cy']:.0f}) d={r['bw']:.0f}x{r['bh']:.0f} szél={r['touches']} "
          f"fény={r['mean']:.1f} tel={r['sat']:.2f}% éles={r['sharp']:.2e} inl={r['inliers']}/{r['matches']} ecc={r['ecc']:.4f} "
          f"sx={r['sx']:.4f} sy={r['sy']:.4f} rot={r['rot']:+.2f}° shear={r['shear']:+.4f} persp~{r['persp_px']:.1f}px")
print("legélesebb:", info[best]["file"])
a = lambda k: np.array([r[k] for r in info], float)
C = lambda u, v: np.corrcoef(u, v)[0, 1]
for k in ("sx", "sy", "rot", "shear", "sharp", "mean"):
    print(f"{k}: ~cx {C(a(k), a('cx')):+.2f} ~cy {C(a(k), a('cy')):+.2f}")

# kontaktlap
th = [cv2.resize(f, (300, 400)) for f in imgs]
while len(th) % 6: th.append(np.zeros_like(th[0]))
sheet = np.vstack([np.hstack(th[i:i+6]) for i in range(0, len(th), 6)])
cv2.imwrite(os.path.join(out, "kontaktlap.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
# 100% kivágás a legélesebbről
r = info[best]; cy, cx = int(r["cy"]), int(r["cx"])
cv2.imwrite(os.path.join(out, "crop_best_100.png"), imgs[best][cy-300:cy+300, cx-300:cx+300])
