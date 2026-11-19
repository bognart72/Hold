"""Képkockánkénti affin transzformáció a legjobb képkockához (ECC): skála, forgás, nyírás.
Plusz 100%-os kivágások a legjobb/legrosszabb képkockából.

Használat: python analyze_geometry.py <video> <elemzes_mappa>
"""
import sys, os, csv
import numpy as np
import cv2

video, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(out, "frames.csv"))))
sharp = np.array([float(r["sharp"]) for r in rows])
best = int(np.argmax(sharp)); worst = int(np.argmin(sharp))
DS = 4

cap = cv2.VideoCapture(video)
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    frames.append(cv2.resize(g, None, fx=1/DS, fy=1/DS, interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    if len(frames) - 1 in (best, worst):
        cv2.imwrite(os.path.join(out, f"full_{len(frames)-1:04d}.png"), f)
cap.release()

ref = cv2.GaussianBlur(frames[best], (0, 0), 1.0)
crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
res = []
warp = np.eye(2, 3, dtype=np.float32)
for i in list(range(best, len(frames))) + list(range(best - 1, -1, -1)):
    if i == best - 1:
        warp = np.eye(2, 3, dtype=np.float32)
    img = cv2.GaussianBlur(frames[i], (0, 0), 1.0)
    try:
        cc, warp = cv2.findTransformECC(ref, img, warp, cv2.MOTION_AFFINE, crit, None, 5)
    except cv2.error:
        cc = np.nan
    A = warp[:, :2].astype(float)
    sx = np.hypot(A[0, 0], A[1, 0]); rot = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    sy = np.linalg.det(A) / sx; shear = (A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / sx**2
    res.append((i, cc, sx, sy, rot, shear, warp[0, 2] * DS, warp[1, 2] * DS))
res.sort()
with open(os.path.join(out, "affine.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["i", "ecc", "sx", "sy", "rot_deg", "shear", "tx", "ty"]); w.writerows(res)
r = np.array(res)
t = np.array([float(x["t"]) for x in rows])
print(f"referencia: #{best}")
print(f"ECC korreláció: min {np.nanmin(r[:,1]):.4f}, medián {np.nanmedian(r[:,1]):.4f}")
for name, col in (("skála x", 2), ("skála y", 3)):
    v = r[:, col]; print(f"{name}: {v.min():.5f}..{v.max():.5f} (szórás {v.std():.5f}) -> Hold-átmérőn ({1700} px) {1700*(v.max()-v.min()):.1f} px")
v = r[:, 4]; p = np.polyfit(t, v, 1)
print(f"forgás: {v.min():.4f}..{v.max():.4f} fok, lineáris trend {p[0]*60:.3f} fok/perc, maradék szórás {np.std(v - np.polyval(p, t)):.4f} fok")
print(f"  -> peremen (r=850 px) a teljes forgás: {np.radians(v.max()-v.min())*850:.1f} px")
v = r[:, 5]; print(f"nyírás: {v.min():.5f}..{v.max():.5f} -> 1700 px-en {1700*(v.max()-v.min()):.1f} px")

# 100% kivágások ugyanarról a részletről
fb = cv2.imread(os.path.join(out, f"full_{best:04d}.png")); fw = cv2.imread(os.path.join(out, f"full_{worst:04d}.png"))
cy, cx = int(float(rows[best]["cy"])), int(float(rows[best]["cx"]))
oy, ox = int(float(rows[worst]["cy"])) - cy, int(float(rows[worst]["cx"])) - cx
y0, x0 = cy - 700, cx - 200
cb = fb[y0:y0+500, x0:x0+500]; cw = fw[y0+oy:y0+oy+500, x0+ox:x0+ox+500]
cv2.imwrite(os.path.join(out, "crop_best_vs_worst.png"), np.hstack([cb, np.full((500, 10, 3), 255, np.uint8), cw]))
