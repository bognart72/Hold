"""RL + védelmek (haló-fék, sötét-kivezetés) régiónkénti hibája a független tesztpadon.
Régiók: perem-sáv (±6 px a korong határán), sötét (a korong mediánjának 30%-a alatt), többi felszín.
Használat: python tools/bench_rl2.py <stack.tif> [...]
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from holdstack import sharpen as S  # noqa: E402
from bench_rl import moon_mask  # noqa: E402  (a bench_rl csak függvényt ad, ha nincs argumentuma)

VARS = [("nincs", None),
        ("régi wavelet", S.DEFAULT),
        ("RL40/0.8", dict(rl_iters=40, rl_frac=0.8, gain=0, shape=.5, k=0, clamp=None)),
        ("+fék 0.25", dict(rl_iters=40, rl_frac=0.8, gain=0, shape=.5, k=0, clamp=0.25)),
        ("+fék 0.1", dict(rl_iters=40, rl_frac=0.8, gain=0, shape=.5, k=0, clamp=0.1)),
        ("+sötét 0.3", dict(rl_iters=40, rl_frac=0.8, gain=0, shape=.5, k=0, clamp=None, dark=0.3)),
        ("+sötét 0.5", dict(rl_iters=40, rl_frac=0.8, gain=0, shape=.5, k=0, clamp=None, dark=0.5)),
        ("+fék 0.25 +sötét 0.3", dict(rl_iters=40, rl_frac=0.8, gain=0, shape=.5, k=0, clamp=0.25, dark=0.3)),
        ("AUTO (sötét 0.3 + peremfék)", S.AUTO)]


def rmse(a, b, sel):
    e = (a - b)[sel]
    return float(np.sqrt(np.mean(e * e)))


tot = {}
for path in sys.argv[1:]:
    L = S.lum(cv2.imread(path, -1).astype(np.float32) / 65535)
    mask = moon_mask(L)
    psf = S.limb_psf_sigma(L, mask)
    T2 = cv2.resize(L, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(mask.astype(np.uint8), (T2.shape[1], T2.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    s_add = float(np.sqrt(max(psf ** 2 - S.limb_psf_sigma(T2, m2) ** 2, 0.09)))
    D2 = cv2.GaussianBlur(T2, (0, 0), s_add) + S.noise_texture(L, mask)[:T2.shape[0], :T2.shape[1]]
    pd = S.limb_psf_sigma(D2, m2)
    sel = S._sel(m2, D2)
    k = np.ones((13, 13), np.uint8)
    band = cv2.dilate(m2.astype(np.uint8), k).astype(bool) & ~cv2.erode(m2.astype(np.uint8), k).astype(bool)
    med = float(np.median(T2[m2]))
    dark = m2 & ~band & (cv2.GaussianBlur(T2, (0, 0), 4) < 0.3 * med)
    rest = m2 & ~band & ~dark
    print(f"\n=== {path}: PSF {pd:.2f}; perem {band.sum()} px, sötét {dark.sum()} px, felszín {rest.sum()} px")
    print(f"  {'változat':24s} {'perem':>7s} {'sötét':>7s} {'felszín':>7s}")
    for name, p in VARS:
        out = D2 if p is None else S.sharpen_L(D2, p, pd, sel, mask=m2)
        r = [rmse(out, T2, s) / med for s in (band, dark, rest)]
        tot.setdefault(name, []).append(r)
        print(f"  {name:24s} " + " ".join(f"{v:7.4f}" for v in r))
print("\nÁtlag (relatív a „nincs”-hez):")
base = np.array(tot["nincs"])
for name, _ in VARS:
    rel = np.array(tot[name]) / base
    print(f"  {name:24s} perem {rel[:, 0].mean():.3f}  sötét {rel[:, 1].mean():.3f}  felszín {rel[:, 2].mean():.3f}")
