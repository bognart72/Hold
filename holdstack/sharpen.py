"""Élesítés v2 (Hold): zajtűrő wavelet, perem-PSF-es dekonvolúció, haló-fék, önkalibráció.

Elvek
-----
* Csak a fényesség élesedik, a színarány marad (a színzaj nem erősödik).
* Zajtűrő wavelet: rétegenként a réteg ERŐSÍTÉSI része (a többlet, (a-1)·d) megy át egy
  „garrote” zsugorításon (d·max(0, 1 − t²/d²), t = k·σ_réteg), az eredeti réteg érintetlen.
  a=1-nél tehát a kép változatlan; a zajszint körüli apró ingadozások nem erősödnek, a
  valódi részletek (|d| ≫ t) teljes erővel igen.
* Dekonvolúció: a Hold megvilágított pereme éles lépcső, a stacken látszó elmosódottsága a
  teljes PSF. Ebből Gauss-szigmát becslünk, és csillapított Richardson–Lucy fut (opcionális).
* Haló-fék: az eredmény a bemenet helyi (5×5) minimuma/maximuma körüli sávba van szorítva
  (a sáv a helyi tartomány m-szerese); a perem és a terminátor menti sötét/világos gyűrű így
  nem nőhet nagyra, a sima részeken a részlet szabadon erősödhet.
* Önkalibráció: a stack saját PSF-jével és saját zajával lerontott változatán keressük azt a
  beállítást, amely a stacket a legjobban visszaadja (RMSE a korongon + a perem sávján).
  A képstatisztika közel skálafüggetlen, így ez a stack → éles kép lépésre is jó becslés.
"""
import numpy as np
import cv2

from .sunpost import rl_deconv

DEFAULT = dict(rl_iters=0, rl_frac=0.9, gain=1.0, shape=0.5, k=0.0, clamp=None)


def lum(img):
    return img if img.ndim == 2 else (0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2])


def amounts(gain, shape, n=4):
    """Rétegerősségek: a_i = 1 + gain·shape^i (i = 0 a legfinomabb). gain=1, shape=0.5 ≈ a régi alap."""
    return tuple(1.0 + gain * shape ** i for i in range(n))


# ---------------------------------------------------------------------------------- PSF a peremből
def fit_limb(mask, tol=1.5, iters=400, seed=0):
    """Kör a megvilágított peremre (RANSAC a maszk kontúrján; a terminátor nem kör, kiesik).
    Visszaad: (cx, cy, R, inlier pontok) vagy None."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    P = max(cnts, key=len)[:, 0, :].astype(np.float64)
    H, W = mask.shape
    P = P[(P[:, 0] > 3) & (P[:, 0] < W - 4) & (P[:, 1] > 3) & (P[:, 1] < H - 4)]
    if len(P) < 60:
        return None
    rng = np.random.default_rng(seed)

    def fit(pts):
        A = np.c_[2 * pts, np.ones(len(pts))]
        c, *_ = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)
        return c[0], c[1], np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0))

    best = None
    for _ in range(iters):
        cx, cy, r = fit(P[rng.choice(len(P), 3, replace=False)])
        if not np.isfinite(r) or r < 20:
            continue
        inl = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 60:
        return None
    cx, cy, r = fit(P[best])
    inl = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) < tol
    return cx, cy, r, P[inl]


def edge_sigma(L, cx, cy, R, ang, half=10.0):
    """Elmosott lépcső (erf) illesztése a peremprofilokra. A profilokat szögenként a legmeredekebb
    pontjukra igazítjuk és normáljuk, a mediánjukra illesztünk (szabad belső/külső szinttel).
    A széles „szoknya” (szórt fény) a szabad szintekbe kerül, a szigma a mag szélességét adja."""
    from scipy.optimize import curve_fit
    from scipy.special import erfc
    step = 0.25
    rr = np.arange(-2 * half, 2 * half + 0.01, step)
    xs = (cx + (R + rr[None, :]) * np.cos(ang[:, None])).astype(np.float32)
    ys = (cy + (R + rr[None, :]) * np.sin(ang[:, None])).astype(np.float32)
    prof = cv2.remap(L.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sm = cv2.GaussianBlur(prof, (0, 0), sigmaX=2.0, sigmaY=0.01)
    n = int(half / step)
    al = []
    for p, s_ in zip(prof, sm):
        i = int(np.argmin(np.gradient(s_)))
        if i < n or i > len(rr) - n - 1:
            continue
        seg = p[i - n:i + n + 1]
        hi, lo = np.median(seg[: n // 3]), np.median(seg[-n // 3:])
        if hi - lo <= 1e-3:
            continue
        al.append((seg - lo) / (hi - lo))
    if len(al) < 40:
        return None, len(al)
    y = np.median(np.array(al), axis=0)
    x = np.arange(-n, n + 1) * step

    def model(x, a, b, x0, s):
        return b + a * 0.5 * erfc((x - x0) / (np.sqrt(2) * s))
    try:
        (a, b, x0, s), _ = curve_fit(model, x, y, p0=(1, 0, 0, 1.5), bounds=([0.3, -0.5, -3, 0.2], [2, 0.5, 3, 9]))
    except Exception:
        return None, len(al)
    return float(s), len(al)


def limb_psf_sigma(L, mask, n_max=720):
    """Gauss-PSF szigma (px) a megvilágított perem profiljából (erf-illesztés); None, ha nem mérhető."""
    f = fit_limb(mask)
    if f is None:
        return None
    cx, cy, R, P = f
    ang = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
    ang = np.sort(ang)[:: max(1, len(ang) // n_max)]
    # a maszk határa fényudvarnál kint lehet az égen: a sugár a medián profil legmeredekebb pontjára
    rr = np.arange(-60, 60.1, 0.5)
    xs = (cx + (R + rr[None, :]) * np.cos(ang[:, None])).astype(np.float32)
    ys = (cy + (R + rr[None, :]) * np.sin(ang[:, None])).astype(np.float32)
    med = np.median(cv2.remap(L.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), 0)
    med = cv2.GaussianBlur(med[None, :].astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=0.01)[0]
    R = R + float(rr[int(np.argmin(np.gradient(med)))])
    s, _ = edge_sigma(L, cx, cy, R, ang)
    return s if s is not None and 0.4 <= s <= 8 else None


# ---------------------------------------------------------------------------------- élesítés
def layers(L, n=4, base=1.0):
    ds, cur = [], L
    for i in range(n):
        g = cv2.GaussianBlur(cur, (0, 0), base * 2 ** i)
        ds.append(cur - g)
        cur = g
    return ds, cur


def layer_noise(ds, sel):
    """Rétegenkénti robusztus szórás (MAD) a kijelölt pixeleken."""
    return [1.4826 * float(np.median(np.abs(d[sel]))) + 1e-7 for d in ds]


def nr_wavelet(L, amts, k=0.0, sig=None, sel=None):
    ds, res = layers(L, len(amts))
    if k > 0 and sig is None:
        sig = layer_noise(ds, sel if sel is not None else np.ones(L.shape, bool))
    out = res.copy()
    for i, (d, a) in enumerate(zip(ds, amts)):
        if k > 0:
            t2 = (k * sig[i]) ** 2
            boost = d * np.clip(1.0 - t2 / np.maximum(d * d, 1e-12), 0, None)
        else:
            boost = d
        out += d + (a - 1.0) * boost
    return out


def clamp_halo(out, ref, m, r=2):
    """A kimenet a bemenet helyi min/max-a ± m·(helyi tartomány) sávjában (puha szorítás)."""
    if m is None:
        return out
    ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    lo, hi = cv2.erode(ref, ker), cv2.dilate(ref, ker)
    pad = m * (hi - lo) + 1e-4
    top, bot = hi + pad, lo - pad
    over = out > top
    under = out < bot
    y = out.copy()
    # a sávon túli rész tanh-sal, legfeljebb még egy „pad”-nyit léphet túl
    y[over] = top[over] + pad[over] * np.tanh((out[over] - top[over]) / pad[over])
    y[under] = bot[under] - pad[under] * np.tanh((bot[under] - out[under]) / pad[under])
    return y


def sharpen_L(L, p, psf=None, sel=None, rl_cache=None, mask=None):
    """Fényesség-csatorna élesítése a p paraméterkészlettel (lásd DEFAULT)."""
    base = L
    if p.get("rl_iters", 0) and psf:
        key = (p["rl_iters"], p.get("rl_frac", 0.9))
        if rl_cache is not None and key in rl_cache:
            base = rl_cache[key]
        else:
            base = rl_deconv(np.clip(L, 0, None) + 1e-4, psf * key[1], key[0]) - 1e-4
            if rl_cache is not None:
                rl_cache[key] = base
    dark = p.get("dark", 0.0)
    if dark and base is not L:
        # sötét területeken (terminátor, árnyék) a dekonvolúció főleg a zajt erősítené: fokozatos
        # visszavezetés az eredetibe, ahol a helyi fényesség a korong mediánjának dark-szorosa alatt van
        ref = float(np.median(L[sel])) if sel is not None and sel.any() else float(np.median(L))
        loc = cv2.GaussianBlur(L, (0, 0), 4)
        w = np.clip((loc - 0.5 * dark * ref) / (0.5 * dark * ref), 0, 1)
        base = w * base + (1 - w) * L
    out = nr_wavelet(base, amounts(p["gain"], p["shape"]), p.get("k", 0.0), sel=sel)
    out = clamp_halo(out, L, p.get("clamp"))
    ec = p.get("edge_clamp")
    if ec is not None and mask is not None and psf:
        # haló-fék csak a korong határának sávjában (perem, terminátor-perem): ott a nagy lépcső
        # gyűrűzik, a felszín belsejében a részlet szabadon maradhat
        m8 = mask.astype(np.uint8)
        # távolság a korong határától (belül az egyik, kívül a másik tag nulla)
        dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3) + cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
        w = np.clip((6 * psf - dist) / (3 * psf), 0, 1)
        out = w * clamp_halo(out, L, ec) + (1 - w) * out
    return out


# a mérésekkel igazolt alapbeállítás (tools/bench_rl.py, bench_rl2.py): RL 40 iteráció a perem-PSF
# 0,8-szeresével, sötét-kivezetés 0,3, haló-fék 0,25 csak a perem sávjában, wavelet nélkül
AUTO = dict(rl_iters=40, rl_frac=0.8, gain=0.0, shape=0.5, k=0.0, clamp=None, dark=0.3, edge_clamp=0.25)


def apply(img, p, psf=None, mask=None):
    """Színes/fekete-fehér kép élesítése; a színarány marad. A bemenet lineáris [0,1]."""
    im = img.astype(np.float32)
    L = lum(im)
    sel = _sel(mask, L)
    Ls = sharpen_L(L, p, psf, sel, mask=mask)
    if im.ndim == 2:
        return Ls
    ratio = (np.maximum(Ls, 0) + 1e-4) / (L + 1e-4)
    return im * ratio[..., None]


def _sel(mask, L):
    if mask is None:
        return np.ones(L.shape, bool)
    s = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    return s if s.sum() > 1000 else mask.astype(bool)


# ---------------------------------------------------------------------------------- önkalibráció
def noise_texture(L, mask, size=96, rng=None):
    """A stack saját zaja: a legsimább, korongon belüli blokk nagyfrekvenciás maradéka,
    véletlen tükrözésekkel/eltolásokkal a teljes képre csempézve (a korreláció megmarad)."""
    rng = rng or np.random.default_rng(1)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    g = cv2.GaussianBlur(L, (0, 0), 3)
    gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    best, bv = None, np.inf
    H, W = L.shape
    for y in range(0, H - size, size // 2):
        for x in range(0, W - size, size // 2):
            if inner[y:y + size, x:x + size].mean() < 0.999:
                continue
            v = float(gm[y:y + size, x:x + size].mean())
            if v < bv:
                best, bv = (y, x), v
    if best is None:
        return None
    y, x = best
    blk = L[y:y + size, x:x + size]
    res = blk - cv2.GaussianBlur(blk, (0, 0), 4)
    res = res - res.mean()
    out = np.zeros_like(L)
    for ty in range(0, H, size):
        for tx in range(0, W, size):
            t = res
            if rng.random() < 0.5:
                t = t[::-1]
            if rng.random() < 0.5:
                t = t[:, ::-1]
            if rng.random() < 0.5:
                t = t.T
            h, w = min(size, H - ty), min(size, W - tx)
            out[ty:ty + h, tx:tx + w] = t[:h, :w]
    return out


def score(out, truth, mask):
    """RMSE a korongon és a perem ±10 px-es sávjában (a haló is büntetve), a korong átlagához."""
    band = cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    e = (out - truth)[band]
    return float(np.sqrt(np.mean(e * e)) / max(float(truth[mask].mean()), 1e-6))


GRID = dict(rl_iters=(0, 10, 20, 40), gain=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
            shape=(0.35, 0.55, 0.75), k=(0.0, 1.0, 2.0, 3.0), clamp=(None, 0.25))


def search(D, T, mask, psf, grid=GRID, log=None):
    """Rácskeresés: D (lerontott) → T (igazság). Visszaad: (legjobb p, pontszám, összes eredmény)."""
    sel = _sel(mask, D)
    cache = {}
    res = []
    for it in grid["rl_iters"]:
        if it and not psf:
            continue
        for g in grid["gain"]:
            for sh in (grid["shape"] if g > 0 else (0.5,)):
                for k in (grid["k"] if g > 0 else (0.0,)):
                    for c in grid["clamp"]:
                        p = dict(rl_iters=it, rl_frac=0.9, gain=g, shape=sh, k=k, clamp=c)
                        res.append((score(sharpen_L(D, p, psf, sel, cache), T, mask), p))
    res.sort(key=lambda r: r[0])
    return res[0][1], res[0][0], res


def pick_window(mask, size=1024):
    """Kalibrációs ablak: a korong és a perem is benne legyen (lefedettség ~70%)."""
    H, W = mask.shape
    if H <= size and W <= size:
        return 0, 0, H, W
    s = min(size, H, W)
    best, bv = (0, 0), -1
    ii = cv2.integral(mask.astype(np.uint8))
    for y in range(0, H - s + 1, s // 8):
        for x in range(0, W - s + 1, s // 8):
            cov = (ii[y + s, x + s] - ii[y, x + s] - ii[y + s, x] + ii[y, x]) / (s * s)
            v = -abs(cov - 0.7)
            if v > bv:
                best, bv = (y, x), v
    y, x = best
    return y, x, y + s, x + s


def calibrate(img, mask, log=None, grid=GRID):
    """Önkalibráció a stackon. Visszaad: (p, info)."""
    L = lum(img.astype(np.float32))
    psf = limb_psf_sigma(L, mask)
    info = dict(psf_sigma=psf)
    if psf is None:
        info["msg"] = "a perem nem mérhető — alapértelmezett élesítés"
        return dict(DEFAULT), info
    y0, x0, y1, x1 = pick_window(mask)
    T = L[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1].astype(bool)
    nz = noise_texture(T, m)
    D = cv2.GaussianBlur(T, (0, 0), psf) + (nz if nz is not None else 0)
    p, sc, res = search(D, T, m, psf, grid)
    base = [r for r in res if r[1]["rl_iters"] == 0 and r[1]["gain"] == 0][0][0]
    old = score(sharpen_L(D, dict(DEFAULT), psf), T, m)
    info.update(window=[int(y0), int(x0), int(y1), int(x1)], best=p, score=sc, score_none=base,
                score_default=old, tried=len(res))
    return p, info
