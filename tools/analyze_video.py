"""Hold-videó diagnosztika: drift, méret, fényesség, élesség, telítettség képkockánként.

Használat: python analyze_video.py <video> <kimeneti_mappa>
"""
import sys, os, csv
import numpy as np
import cv2

video, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cap = cv2.VideoCapture(video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
DS = 4  # elemzés 1/4 felbontáson (960x540)

rows, thumbs, prev = [], {}, None
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    g_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sat_frac = float((g_full >= 250).mean())
    g = cv2.resize(g_full, None, fx=1 / DS, fy=1 / DS, interpolation=cv2.INTER_AREA).astype(np.float32)
    thr, mask = cv2.threshold(g.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # legnagyobb komponens = Hold
    nlab, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    if nlab > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = stats[k]
        cx, cy = cents[k]
        moon = lab == k
        H, W = g.shape
        touches = int(x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1)
        mean_b = float(g[moon].mean())
        # zajtűrő élesség: Gauss után Laplace-variancia a Hold belsejében, fényességgel normálva
        gb = cv2.GaussianBlur(g_full.astype(np.float32), (0, 0), 1.5)
        lap = cv2.Laplacian(gb, cv2.CV_32F)
        moon_full = cv2.resize(moon.astype(np.uint8), (g_full.shape[1], g_full.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        moon_full = cv2.erode(moon_full.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
        sharp = float(lap[moon_full].var() / max(gb[moon_full].mean(), 1) ** 2) if moon_full.any() else 0.0
    else:
        x = y = w = h = area = 0; cx = cy = np.nan; touches = 1; mean_b = 0.0; sharp = 0.0
    # képkocka-képkocka eltolás (fázis-korreláció, 1/4 felbontás)
    win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
    if prev is not None:
        (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
    else:
        dx = dy = 0.0; resp = 1.0
    prev = g
    rows.append(dict(i=i, t=round(t, 4), cx=cx * DS, cy=cy * DS, bw=w * DS, bh=h * DS, area=area * DS * DS,
                     touches=touches, mean=mean_b, sat=sat_frac, sharp=sharp,
                     dx=dx * DS, dy=dy * DS, pc_resp=resp))
    if i % 100 == 0 or i in (1, 2):
        thumbs[i] = frame
    i += 1
cap.release()

keys = list(rows[0].keys())
with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)

a = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
order = np.argsort(-a["sharp"])
best, worst = int(order[0]), int(order[-1])
print(f"képkockák: {len(rows)} (fejléc szerint {n})")
dt = np.diff(a["t"]); print(f"dt: min {dt.min()*1000:.1f} ms, max {dt.max()*1000:.1f} ms, medián {np.median(dt)*1000:.1f} ms, >40ms ugrás: {(dt>0.04).sum()}")
print(f"Hold-középpont x: {np.nanmin(a['cx']):.0f}..{np.nanmax(a['cx']):.0f}, y: {np.nanmin(a['cy']):.0f}..{np.nanmax(a['cy']):.0f}")
print(f"teljes drift: dx={np.nansum(a['dx']):.0f} px, dy={np.nansum(a['dy']):.0f} px; max ugrás képkockák között: {np.hypot(a['dx'],a['dy']).max():.1f} px")
print(f"befoglaló doboz w: {a['bw'].min():.0f}..{a['bw'].max():.0f}, h: {a['bh'].min():.0f}..{a['bh'].max():.0f}")
print(f"szélhez ér: {int(a['touches'].sum())} képkocka")
print(f"átlagfényesség (Hold): {a['mean'].min():.1f}..{a['mean'].max():.1f} (szórás {a['mean'].std():.2f})")
print(f"telített pixelarány max: {a['sat'].max()*100:.3f}%")
print(f"élesség: min {a['sharp'].min():.3e}, medián {np.median(a['sharp']):.3e}, max {a['sharp'].max():.3e}; legjobb #{best}, legrosszabb #{worst}")
print(f"fázis-korreláció válasz min: {a['pc_resp'][1:].min():.3f}")

cap = cv2.VideoCapture(video)
for name, idx in (("best", best), ("worst", worst)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read()
    if ok: cv2.imwrite(os.path.join(out, f"{name}_{idx:04d}.png"), fr)
for idx, fr in thumbs.items():
    cv2.imwrite(os.path.join(out, f"frame_{idx:04d}.jpg"), cv2.resize(fr, None, fx=0.25, fy=0.25), [cv2.IMWRITE_JPEG_QUALITY, 85])
