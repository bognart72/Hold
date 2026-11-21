"""Szintetikus napfelvétel ismert „igazsággal” (ground truth) a Nap-mód teszteléséhez.

Tartalom: szélsötétedéses korong, granuláció, napfoltok (umbra + penumbra), fáklyamezők a perem
közelében; képkockánként: drift (nem követő Dobson), lassú mezőrotáció, szenzorhoz kötött torzítás
és vinyettálás, SZENZORHOZ KÖTÖTT PORSZEMEK (napfoltnak látszó sötét pöttyök!), seeing-warp,
változó elmosódás (lucky imaging), zaj, narancs napszűrő-szín, JPEG.

Használat: python tools/make_sun_test.py <kimeneti_mappa> [--frames 40]
Kimenet: frame_XXX.jpg, ground_truth.png (a Nap a 0. kocka geometriájában, torzítás nélkül),
truth.json (foltok és porszemek adatai).
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--frames", type=int, default=40)
ap.add_argument("--size", type=int, default=2400)
ap.add_argument("--diam", type=int, default=1700)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--gran", type=float, default=0.025, help="granuláció amplitúdója")
ap.add_argument("--drift", type=float, default=1.0, help="drift szorzó (1 = ~340 px)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
# az igazság-fájlok külön mappába (különben a stackelő bemeneti képként olvasná őket)
truth_dir = os.path.normpath(a.out) + "_truth"
os.makedirs(truth_dir, exist_ok=True)
rng = np.random.default_rng(a.seed)
S, D = a.size, a.diam
R = D / 2.0

# ---- a Nap felszíne (saját koordinátákban, középpont = (S/2, S/2)) -----------------------------
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
cx = cy = S / 2.0
r = np.hypot(xx - cx, yy - cy) / R
inside = r < 1.0
mu = np.sqrt(np.clip(1 - r ** 2, 0, 1))
sun = (1 - 0.62 * (1 - mu) - 0.08 * (1 - mu) ** 2).astype(np.float32)  # szélsötétedés
gran = cv2.GaussianBlur(rng.standard_normal((S, S)).astype(np.float32), (0, 0), 2.2)
gran /= gran.std()
sun *= 1 + a.gran * gran
# fáklyamezők a perem közelében
for _ in range(10):
    ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.82, 0.96) * R
    fx, fy = cx + rr * np.cos(ang), cy + rr * np.sin(ang)
    blob = np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * rng.uniform(15, 40) ** 2))
    sun *= 1 + 0.08 * blob
spots = []
for k in range(9):
    while True:
        ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.0, 0.8) * R
        sx, sy = cx + rr * np.cos(ang), cy + rr * np.sin(ang)
        if all(np.hypot(sx - s["x"], sy - s["y"]) > 120 for s in spots):
            break
    ru = float(rng.choice([2.5, 4, 6, 9, 14]))          # umbra sugara: a legkisebbek „pici” foltok
    rp = ru * rng.uniform(1.8, 2.6)
    d2 = np.hypot(xx - sx, yy - sy)
    umbra = 1 - 0.78 * np.clip((ru + 1 - d2) / 2, 0, 1)
    pen = 1 - 0.32 * np.clip((rp + 1.5 - d2) / 3, 0, 1)
    sun *= np.minimum(umbra, pen)
    spots.append(dict(x=float(sx), y=float(sy), r_umbra=ru, r_penumbra=float(rp)))
sun = np.where(inside, sun, 0).astype(np.float32)
# a perem enyhe elmosása (anti-alias)
sun = cv2.GaussianBlur(sun, (0, 0), 0.6)
cv2.imwrite(os.path.join(truth_dir, "ground_truth.png"), np.clip(sun * 255 / sun.max(), 0, 255).astype(np.uint8))

# ---- szenzorhoz kötött hibák ---------------------------------------------------------------------
H = W = S
sy_, sx_ = np.mgrid[0:H, 0:W].astype(np.float32)
nx, ny = (sx_ - W / 2) / (W / 2), (sy_ - H / 2) / (H / 2)
vign = (1 - 0.12 * ((nx - 0.25) ** 2 + (ny + 0.1) ** 2)).astype(np.float32)      # nem tengelyes vinyettálás
dust = np.ones((H, W), np.float32)
dusts = []
for _ in range(6):
    dx, dy = rng.uniform(0.3 * W, 0.7 * W), rng.uniform(0.3 * H, 0.7 * H)
    rd = rng.uniform(3, 8); depth = rng.uniform(0.25, 0.45)
    dust *= 1 - depth * np.exp(-((sx_ - dx) ** 2 + (sy_ - dy) ** 2) / (2 * (rd / 1.6) ** 2))
    dusts.append(dict(x=float(dx), y=float(dy), r=float(rd), depth=float(depth)))
# szenzorhoz kötött torzítás: u = x + k * (2. fokú tag)
def distort_map(ux, uy):
    a_ = (ux - W / 2) / (W / 2); b_ = (uy - H / 2) / (H / 2)
    return ux + 6.0 * (a_ * a_ - 0.4 * a_ * b_), uy + 6.0 * (0.5 * b_ * b_ + 0.3 * a_ * b_)

color = np.array([0.30, 0.68, 1.0], np.float32)  # BGR: narancs napszűrő
meta = []
for k in range(a.frames):
    t = k / max(a.frames - 1, 1)
    # drift (a Nap vándorol a képen), lassú forgás
    tx, ty = a.drift * (-150 + 300 * t) + rng.normal(0, 2), a.drift * (-90 + 180 * t) + rng.normal(0, 2)
    rot = np.radians(0.4 * t)
    blur = float(rng.choice([1.0, 1.4, 1.8, 2.4, 3.0], p=[0.15, 0.25, 0.25, 0.2, 0.15]))
    # seeing: sima véletlen eltolási mező
    wx = cv2.GaussianBlur(rng.standard_normal((H // 16, W // 16)).astype(np.float32), (0, 0), 2.5)
    wy = cv2.GaussianBlur(rng.standard_normal((H // 16, W // 16)).astype(np.float32), (0, 0), 2.5)
    wx = cv2.resize(wx / (wx.std() + 1e-6), (W, H)) * 1.2 * blur / 2
    wy = cv2.resize(wy / (wy.std() + 1e-6), (W, H)) * 1.2 * blur / 2
    # szenzor-pixel -> ideális szenzor (torzítás) -> Nap-koordináta (forgás + eltolás) + seeing
    ux, uy = distort_map(sx_, sy_)
    c, s_ = np.cos(rot), np.sin(rot)
    px = c * (ux - W / 2 - tx) + s_ * (uy - H / 2 - ty) + cx + wx
    py = -s_ * (ux - W / 2 - tx) + c * (uy - H / 2 - ty) + cy + wy
    img = cv2.remap(sun, px.astype(np.float32), py.astype(np.float32), cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img * vign * dust * 0.88
    img = img + rng.normal(0, 0.012, img.shape).astype(np.float32) * np.sqrt(np.clip(img, 0, 1))
    img = np.clip(img + 0.01, 0, 1)
    bgr = (img[..., None] * color[None, None, :])
    bgr = np.clip(bgr * 255 + rng.normal(0, 0.6, bgr.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(a.out, f"frame_{k:03d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    meta.append(dict(frame=k, tx=float(tx), ty=float(ty), rot_deg=float(np.degrees(rot)), blur=blur))
    print(f"kocka {k + 1}/{a.frames} (elmosás {blur})", flush=True)

with open(os.path.join(truth_dir, "truth.json"), "w", encoding="utf-8") as fh:
    json.dump(dict(size=S, diam=D, spots=spots, dust=dusts, frames=meta,
                   note="ground_truth.png: a Nap saját koordinátáiban (középpont S/2,S/2), a 0. kocka "
                        "forgatása 0; a kockák ehhez képest eltoltak/forgatottak"), fh, ensure_ascii=False, indent=1)
print("kész")

