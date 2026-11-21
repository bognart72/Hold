"""Utófeldolgozás: többszintű (Registax/PSS-stílusú) wavelet-élesítés, fényességkezelés, mentés.

Fényesség-elv: a stack súlyozott ÁTLAG, ezért az eredmény fényessége megegyezik a bemenetével.
Semmilyen automatikus széthúzás nincs (csak kérésre), az élesítés túllövéseit pedig lágy
csúcsfény-levágás (roll-off) kezeli, hogy a fényes részek ne égjenek be.
"""
import numpy as np
import cv2


def wavelet_sharpen(img, amounts=(2.0, 1.5, 1.2, 1.0), base_sigma=1.0, luminance_only=True):
    """Gauss-piramisos részletrétegek erősítése. amount=1 -> változatlan réteg.
    Színes képnél csak a fényességet élesíti (a színzaj nem erősödik), a színarány megmarad."""
    img = img.astype(np.float32)
    if luminance_only and img.ndim == 3:
        L = 0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2]
        Ls = wavelet_sharpen(L, amounts, base_sigma, False)
        ratio = (np.maximum(Ls, 0) + 1e-4) / (L + 1e-4)
        return img * ratio[..., None]
    layer = img
    out = np.zeros_like(layer)
    for i, a in enumerate(amounts):
        g = cv2.GaussianBlur(layer, (0, 0), base_sigma * (2 ** i))
        out += a * (layer - g)
        layer = g
    return out + layer


def soft_clip(x, knee=0.85):
    """Lágy csúcsfény-levágás: knee alatt lineáris, felette tanh-görbével közelít 1-hez."""
    y = x.copy()
    hi = x > knee
    r = 1.0 - knee
    y[hi] = knee + r * np.tanh((x[hi] - knee) / r)
    return np.clip(y, 0.0, 1.0)


def stretch(img, mask, black_pct=0.05, white_pct=99.9, white_level=0.92):
    """Opcionális széthúzás fej-térrel: a white_pct percentilis white_level-re kerül (nem 1-re)."""
    lo = np.percentile(img, black_pct)
    sel = img[mask] if mask is not None and mask.any() else img
    hi = np.percentile(sel, white_pct)
    return (img - lo) / max(hi - lo, 1e-6) * white_level


def clipped_fraction(img, mask=None, level=0.98):
    L = img if img.ndim == 2 else (0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2])
    sel = L[mask] if mask is not None and mask.any() else L
    return float((sel >= level).mean())


def sun_enhance(img, mask, bins=256, gain=1.6):
    """Nap: a szélsötétedés (limb darkening) kiegyenlítése és enyhe helyi kontrasztkiemelés.

    A korong fényességének sugárirányú profilját (a foltokat kizáró, felső-medián statisztikával)
    kiszámolja és leosztja vele a képet, így a perem közeli fáklyamezők és a foltok egyforma
    láthatóságúak lesznek. Utána finom (σ≈30 px) helyi kontraszterősítés. A színarány marad.
    Ez megjelenítési változat; a mérésre/archiválásra a _stack.tif való."""
    im = img.astype(np.float32)
    L = im if im.ndim == 2 else (0.0722 * im[..., 0] + 0.7152 * im[..., 1] + 0.2126 * im[..., 2])
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return np.clip(im, 0, 1)
    cx, cy = xs.mean(), ys.mean()
    R = np.sqrt(len(xs) / np.pi)
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    r = np.hypot(xx - cx, yy - cy) / R
    idx = np.clip((r * bins).astype(int), 0, bins - 1)
    prof = np.zeros(bins, np.float32)
    for b in range(bins):
        v = L[(idx == b) & mask]
        prof[b] = np.percentile(v, 60) if v.size > 20 else np.nan
    good = np.isfinite(prof)
    prof = np.interp(np.arange(bins), np.nonzero(good)[0], prof[good]).astype(np.float32)
    prof = cv2.GaussianBlur(prof[None, :], (0, 0), 2)[0]
    # a kiegyenlítés erőssége korlátozott (a legszélső, nagyon sötét sávban az osztás zajt erősítene),
    # és a perem utolsó ~3%-ában fokozatosan elhalványul
    pc = float(np.median(prof[: bins // 4]))
    gainmap = pc / np.maximum(prof, pc / 2.2)
    fade = np.clip((0.995 - r) / 0.03, 0, 1)
    corr = 1 + (gainmap[idx] - 1) * fade
    flat = np.where(mask, L * corr / pc, 0)
    # helyi kontraszt: a részlet erősítése a környezethez képest
    # normált konvolúció: a korongon kívüli sötét ég ne húzza le a peremnél a helyi átlagot
    mf = mask.astype(np.float32)
    base = cv2.GaussianBlur(flat * mf, (0, 0), 30) / np.maximum(cv2.GaussianBlur(mf, (0, 0), 30), 1e-3)
    det = flat - base
    out = np.where(mask, base + gain * det, 0)
    med = float(np.median(out[mask]))
    out = out * (0.72 / max(med, 1e-4))
    ratio = out / np.maximum(L, 1e-4)
    res = im * ratio[..., None] if im.ndim == 3 else out
    return soft_clip(np.clip(res, 0, None).astype(np.float32), 0.85)


def save16(path, img):
    cv2.imwrite(path, np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16))


def save8(path, img):
    cv2.imwrite(path, np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
