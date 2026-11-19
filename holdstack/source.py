"""Egységes bemenet: videófájl vagy képmappa (EXIF-forgatással és -időbélyeggel)."""
import os
import glob
from datetime import datetime

import numpy as np
import cv2
from PIL import Image, ImageOps

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def to_float(raw):
    """Nyers képkocka (uint8/uint16 BGR) -> float32 [0,1]."""
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) * (1.0 / 255.0)
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) * (1.0 / 65535.0)
    return raw.astype(np.float32)


def luminance(f):
    """BGR float -> fénysűrűség (Rec.709 súlyok)."""
    if f.ndim == 2:
        return f
    return cv2.transform(f, np.array([[0.0722, 0.7152, 0.2126]], np.float32))


def _exif_time(path):
    try:
        im = Image.open(path)
        ex = im.getexif().get_ifd(0x8769)
        dt = ex.get(36867)
        sub = ex.get(37521) or "0"
        if dt:
            t = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").timestamp()
            return t + float("0." + str(sub).strip())
    except Exception:
        pass
    return None


def _load_image(path):
    if path.lower().endswith((".tif", ".tiff", ".png")):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 16 bit megtartása
        if raw is not None:
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            elif raw.shape[2] == 4:
                raw = raw[..., :3]
            # EXIF-forgatás csak JPEG-nél szokásos; PNG/TIFF esetén nem forgatunk
            return raw
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


class Source:
    """Képkockák forrása. `frames(indices)` sorban adja vissza a kért képkockákat (index, nyers BGR)."""

    def __init__(self, path):
        self.path = path
        if os.path.isdir(path):
            self.kind = "images"
            files = [p for p in glob.glob(os.path.join(path, "*")) if p.lower().endswith(IMG_EXT)]
            if not files:
                raise ValueError(f"Nincs kép a mappában: {path}")
            times = [_exif_time(p) for p in files]
            if all(t is not None for t in times):
                order = np.argsort(times)
                self.files = [files[i] for i in order]
                t0 = min(times)
                self.timestamps = [times[i] - t0 for i in order]
            else:
                self.files = sorted(files)
                self.timestamps = [float(i) for i in range(len(self.files))]
            self.n = len(self.files)
            first = _load_image(self.files[0])
            self.shape = first.shape[:2]
        else:
            self.kind = "video"
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Nem nyitható meg: {path}")
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, f = cap.read()
            if not ok:
                raise ValueError("Üres videó")
            self.shape = f.shape[:2]
            cap.release()
            self.timestamps = None  # az első teljes olvasáskor töltődik

    def names(self):
        if self.kind == "images":
            return [os.path.basename(p) for p in self.files]
        return [f"frame_{i:05d}" for i in range(self.n)]

    def frames(self, indices=None):
        want = None if indices is None else set(int(i) for i in indices)
        if self.kind == "images":
            for i, p in enumerate(self.files):
                if want is None or i in want:
                    raw = _load_image(p)
                    if raw.shape[:2] != self.shape:
                        raise ValueError(f"Eltérő képméret: {p} {raw.shape[:2]} != {self.shape}")
                    yield i, raw
            return
        cap = cv2.VideoCapture(self.path)
        ts = []
        i = 0
        last = max(want) if want else None
        while True:
            if want is not None and i in want or want is None:
                ok, raw = cap.read()
                if not ok:
                    break
                ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                yield i, raw
            else:
                if not cap.grab():
                    break
                ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            i += 1
            if last is not None and i > last:
                break
        cap.release()
        if want is None:
            self.timestamps = ts
            self.n = len(ts)
