"""Hold-élesítés tesztpad: önkalibráció vs. független „igazság”.

Használat: python tools/bench_sharpen.py <stack.tif> [...]

Minden stackra:
  1. önkalibráció a teljes felbontású stackon (holdstack.sharpen.calibrate)
  2. független ellenőrzés: igazság T2 = a stack felére kicsinyítve (élesebb, kevésbé zajos);
     D2 = T2 elmosva úgy, hogy a perem-PSF a stackéval egyezzen, + a stack valódi zaja.
     D2-n „vakon” (T2 ismerete nélkül) önkalibrálunk, és összevetjük:
       nincs élesítés | régi alap | önkalibrált | orákulum (T2-höz optimalizált)
"""
import sys
import time
import json

import numpy as np
import cv2

sys.path.insert(0, ".")
from holdstack import sharpen as S  # noqa: E402


def moon_mask(L):
    m = (L > 0.15 * np.percentile(L, 99.5)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    f = np.zeros_like(m)
    cv2.drawContours(f, cnts, -1, 1, -1)
    return f.astype(bool)


def fmt(p):
    return (f"RL{p['rl_iters']:>2} gain {p['gain']:.1f} shape {p['shape']:.2f} k {p['k']:.0f} "
            f"clamp {p['clamp']}")


summary = []
for path in sys.argv[1:]:
    t0 = time.time()
    img = cv2.imread(path, -1).astype(np.float32) / 65535
    L = S.lum(img)
    mask = moon_mask(L)
    print(f"\n=== {path}  {L.shape[1]}x{L.shape[0]}")
    p_self, info = S.calibrate(img, mask)
    print(f"  perem-PSF σ = {info['psf_sigma']}")
    if info.get("best"):
        print(f"  önkalibráció: {fmt(p_self)}  | hiba: nincs {info['score_none']:.4f}, régi {info['score_default']:.4f},"
              f" legjobb {info['score']:.4f}  ({time.time() - t0:.0f} s)")
    psf = info["psf_sigma"]
    # ---- független ellenőrzés
    T2 = cv2.resize(L, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(mask.astype(np.uint8), (T2.shape[1], T2.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    psf2 = S.limb_psf_sigma(T2, m2)
    s_add = float(np.sqrt(max(psf ** 2 - psf2 ** 2, 0.09)))
    nz = S.noise_texture(L, mask)[:T2.shape[0], :T2.shape[1]]
    D2 = cv2.GaussianBlur(T2, (0, 0), s_add) + nz
    print(f"  ellenőrzés: igazság PSF {psf2:.2f} px, +elmosás {s_add:.2f} px, D2 PSF {S.limb_psf_sigma(D2, m2):.2f} px")
    p_blind, ib = S.calibrate(D2, m2)
    _, _, res = S.search(D2, T2, m2, S.limb_psf_sigma(D2, m2))
    p_or, sc_or = res[0][1], res[0][0]
    sc = lambda p: S.score(S.sharpen_L(D2, p, S.limb_psf_sigma(D2, m2), S._sel(m2, D2)), T2, m2)  # noqa: E731
    r = dict(file=path, psf=psf, none=sc(dict(S.DEFAULT, gain=0.0)), old=sc(S.DEFAULT), blind=sc(p_blind),
             oracle=sc_or, p_blind=p_blind, p_oracle=p_or, p_self=p_self)
    print(f"  D2→T2 hiba: nincs {r['none']:.4f} | régi alap {r['old']:.4f} | önkalibrált {r['blind']:.4f} "
          f"| orákulum {r['oracle']:.4f}")
    print(f"    önkalibrált: {fmt(p_blind)}\n    orákulum:    {fmt(p_or)}")
    # a legjobb néhány orákulum-változat (mennyire lapos az optimum)
    for s_, p in res[:5]:
        print(f"      {s_:.4f}  {fmt(p)}")
    summary.append(r)
    print(f"  ({time.time() - t0:.0f} s)")
json.dump(summary, open("eredmeny/_bench_sharpen.json", "w", encoding="utf-8"), indent=1, default=str)
