"""Szelídebb dekonvolúció + csúcsfény-védelem: hiba a független tesztpadon és csúcsfény a valódi stacken.
Használat: python tools/bench_gentle.py <stack.tif> [...]
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from holdstack import sharpen as S  # noqa: E402
from holdstack.post import wavelet_sharpen, soft_clip  # noqa: E402
from bench_rl import moon_mask  # noqa: E402

B = dict(gain=0.0, shape=0.5, k=0.0, clamp=None, dark=0.3, edge_clamp=0.25)
VARS = [("régi wavelet", None),
        ("v6: RL40/0.8", dict(B, rl_iters=40, rl_frac=0.8)),
        ("RL40/0.8 +védelem", dict(B, rl_iters=40, rl_frac=0.8, hi_cap=0.02)),
        ("RL20/0.7 +védelem", dict(B, rl_iters=20, rl_frac=0.7, hi_cap=0.02)),
        ("RL20/0.8 +védelem", dict(B, rl_iters=20, rl_frac=0.8, hi_cap=0.02)),
        ("RL20/0.8 70% +védelem", dict(B, rl_iters=20, rl_frac=0.8, hi_cap=0.02, strength=0.7)),
        ("RL30/0.8 70% +védelem", dict(B, rl_iters=30, rl_frac=0.8, hi_cap=0.02, strength=0.7)),
        ("RL20/0.8 50% +védelem", dict(B, rl_iters=20, rl_frac=0.8, hi_cap=0.02, strength=0.5))]
acc = {}
for path in sys.argv[1:]:
    img = cv2.imread(path, -1).astype(np.float32) / 65535
    L = S.lum(img)
    mask = moon_mask(L)
    psf = S.limb_psf_sigma(L, mask)
    # független tesztpad
    T2 = cv2.resize(L, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    m2 = cv2.resize(mask.astype(np.uint8), (T2.shape[1], T2.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    s_add = float(np.sqrt(max(psf ** 2 - S.limb_psf_sigma(T2, m2) ** 2, 0.09)))
    D2 = cv2.GaussianBlur(T2, (0, 0), s_add) + S.noise_texture(L, mask)[:T2.shape[0], :T2.shape[1]]
    pd = S.limb_psf_sigma(D2, m2)
    sel2 = S._sel(m2, D2)
    none = S.score(D2, T2, m2)
    print(f"\n=== {path}  PSF {psf:.2f}")
    print(f"  {'változat':24s} {'hiba':>6s} | {'>0.95 %':>8s} {'>0.85 %':>8s} {'p99.9':>6s}")
    for name, p in VARS:
        if p is None:
            e = S.score(S.sharpen_L(D2, S.DEFAULT, pd, sel2), T2, m2)
            out = soft_clip(np.maximum(wavelet_sharpen(img), 0), 0.85)
        else:
            e = S.score(S.sharpen_L(D2, p, pd, sel2, mask=m2), T2, m2)
            out = soft_clip(np.maximum(S.apply(img, p, psf, mask), 0), 0.85)
        P = out.max(2)[mask]
        r = (e / none, (P > 0.95).mean() * 100, (P > 0.85).mean() * 100, np.percentile(P, 99.9))
        acc.setdefault(name, []).append(r)
        print(f"  {name:24s} {r[0]:6.3f} | {r[1]:8.3f} {r[2]:8.2f} {r[3]:6.3f}")
print("\nÁtlag:")
for name, _ in VARS:
    a = np.array(acc[name])
    print(f"  {name:24s} hiba {a[:, 0].mean():.3f} | >0.95 {a[:, 1].mean():.3f}%  >0.85 {a[:, 2].mean():.2f}%")
