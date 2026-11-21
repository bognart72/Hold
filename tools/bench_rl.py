"""Célzott mérés: RL-iteráció × PSF-arány (+ utólagos wavelet) a független tesztpadon.

Igazság T2 = a stack felére kicsinyítve; D2 = T2 elmosva a stack perem-PSF-jére + a stack saját zaja.
Használat: python tools/bench_rl.py <stack.tif> [...]
"""
import sys
import numpy as np
import cv2

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from holdstack import sharpen as S  # noqa: E402


def moon_mask(L):
    m = (L > 0.15 * np.percentile(L, 99.5)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    f = np.zeros(L.shape, np.uint8)
    cv2.drawContours(f, cnts, -1, 1, -1)
    return f.astype(bool)


def main():
    ITERS = (10, 20, 30, 40, 60, 80)
    FRACS = (0.7, 0.8, 0.9, 1.0)
    tab = {}
    for path in sys.argv[1:]:
        img = cv2.imread(path, -1).astype(np.float32) / 65535
        L = S.lum(img)
        mask = moon_mask(L)
        psf = S.limb_psf_sigma(L, mask)
        T2 = cv2.resize(L, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(mask.astype(np.uint8), (T2.shape[1], T2.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        s_add = float(np.sqrt(max(psf ** 2 - S.limb_psf_sigma(T2, m2) ** 2, 0.09)))
        D2 = cv2.GaussianBlur(T2, (0, 0), s_add) + S.noise_texture(L, mask)[:T2.shape[0], :T2.shape[1]]
        pd = S.limb_psf_sigma(D2, m2)
        sel = S._sel(m2, D2)
        none = S.score(D2, T2, m2)
        old = S.score(S.sharpen_L(D2, S.DEFAULT, pd, sel), T2, m2)
        print(f"\n=== {path}: PSF {psf:.2f} (D2 {pd:.2f}); nincs {none:.4f}, régi wavelet {old:.4f}")
        print("  iter " + " ".join(f"frac{f:>4}" for f in FRACS))
        for it in ITERS:
            row = []
            for f in FRACS:
                v = S.score(S.sharpen_L(D2, dict(rl_iters=it, rl_frac=f, gain=0.0, shape=0.5, k=0, clamp=None), pd, sel),
                            T2, m2)
                tab.setdefault((it, f), []).append(v / none)
                row.append(v)
            print(f"  {it:>4} " + " ".join(f"{v:.4f} " for v in row))
        # wavelet a dekonvolúció után
        for g, k in ((0.3, 2), (0.5, 2), (0.5, 3), (1.0, 3)):
            v = S.score(S.sharpen_L(D2, dict(rl_iters=30, rl_frac=0.9, gain=g, shape=0.35, k=k, clamp=None), pd, sel),
                        T2, m2)
            print(f"  RL30/0.9 + wavelet gain {g} k {k}: {v:.4f}")
    print("\nÁtlagos relatív hiba (élesítés nélkül = 1):")
    for (it, f), v in sorted(tab.items(), key=lambda kv: np.mean(kv[1]))[:8]:
        print(f"  RL{it:>3} frac {f}: {np.mean(v):.3f}  (max {np.max(v):.3f})")


if __name__ == "__main__":
    main()
