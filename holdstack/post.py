"""Utófeldolgozás: többszintű (Registax/PSS-stílusú) wavelet-élesítés és mentés."""
import numpy as np
import cv2


def wavelet_sharpen(img, amounts=(2.0, 1.5, 1.2, 1.0), base_sigma=1.0):
    """Gauss-piramisos részletrétegek erősítése. amount=1 -> változatlan réteg.
    A rétegek szigmái: base_sigma * 2^i."""
    layer = img.astype(np.float32)
    out = np.zeros_like(layer)
    for i, a in enumerate(amounts):
        g = cv2.GaussianBlur(layer, (0, 0), base_sigma * (2 ** i))
        out += a * (layer - g)
        layer = g
    return out + layer


def save16(path, img):
    cv2.imwrite(path, np.clip(img * 65535.0 + 0.5, 0, 65535).astype(np.uint16))


def save8(path, img):
    cv2.imwrite(path, np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
