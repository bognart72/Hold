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


def save16(path, img):
    cv2.imwrite(path, np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16))


def save8(path, img):
    cv2.imwrite(path, np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
