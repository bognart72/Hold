"""Folyamatalapú (multiprocessing) AP-mérés.

Az AP-nkénti cv2.matchTemplate-ciklus Python-része a GIL-t tartja, ezért szálakkal ~2x-nél nem
gyorsul. Külön folyamatokban viszont valóban párhuzamos. A munkafolyamatok egyszer kapják meg a
közös adatokat (referencia, AP-k, flat, leképezések), utána képkockánként csak a nyers kép megy át.
"""
import numpy as np
import cv2

from .source import to_float, luminance
from . import ap as apmod

_C = {}


def init(ctx):
    cv2.setNumThreads(1)  # a párhuzamosság a folyamatok szintjén van
    _C.clear()
    _C.update(ctx)
    for fm in _C["maps"].values():
        if fm.dist is not None:
            fm.dist.dense_D()  # egyszer folyamatonként (a Distortion-t a folyamatok közösen használják)
            break


def measure_frame(i, raw):
    c = _C
    f = to_float(raw)
    if c["inv_g"] is not None:
        f *= c["inv_g"][..., None]
    g_ = c["gains"].get(i)
    if g_:
        f *= np.float32(np.exp(-g_))
    g = luminance(f)
    fm = c["maps"][i]
    w = fm.warp(g, c["W"], c["H"], c["cg"], cv2.INTER_LINEAR, -1.0)
    valid = w >= 0
    wb = cv2.GaussianBlur(np.where(valid, w, 0).astype(np.float32), (0, 0), 1.0)
    s, o, _ = apmod.measure(wb, valid, c["refb"], c["aps"], c["hb"], c["search"], c["min_corr"])
    pts = fm.points(c["aps"][:, ::-1].astype(np.float64))
    return i, s, o, apmod.local_quality_at(g, pts, c["hb"])
