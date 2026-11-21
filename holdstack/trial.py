"""Próbafuttatásos kalibráció: a legjobb AP-box méret és kiválasztási arány megkeresése
ténylegesen lefuttatott mini-stackekkel.

A közös, drága lépések (elemzés, globális illesztés, flat, torzítás, ROI) egyszer futnak le; a
próbák egy jól strukturált csempén (tile) dolgoznak, a képkockák egy részhalmazán, előre
kivágott, globálisan illesztett szürke csempékből. Minden jelölt (AP-box × arány) valódi
AP-mérést, AP-nkénti kiválasztást és warp-mezős stacket kap, majd pontszámot:

    részlet  S = var(DoG(σ=1) − DoG(σ=3)) / átlag²        (középfrekvenciás, valódi holdrészlet)
    zaj      N = 1.4826·median|I − G(σ=0.7)| / átlag      (legmagasabb frekvencia)
    pontszám = S / √N

A legmagasabb pontszámú beállítással fut tovább a teljes feldolgozás.
"""
import time

import numpy as np
import cv2

from . import geometry as geo
from . import ap as apmod
from .distortion import CoarseGrid


def pick_tile(ref, mask, size):
    """A legtöbb struktúrát tartalmazó size×size-es csempe a Hold belsejében. Visszaad: (y0, x0, h, w)."""
    H, W = ref.shape
    size = int(min(size, H, W))
    b = cv2.GaussianBlur(ref, (0, 0), 1.5)
    e = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
    e = cv2.GaussianBlur(e * geo.erode(mask, 20), (0, 0), size / 5.0)
    h2 = size // 2
    e[:h2, :] = 0; e[-h2:, :] = 0; e[:, :h2] = 0; e[:, -h2:] = 0
    cy, cx = np.unravel_index(int(np.argmax(e)), e.shape)
    y0 = int(np.clip(cy - h2, 0, H - size)); x0 = int(np.clip(cx - h2, 0, W - size))
    return y0, x0, size, size


def score(img, mask, sat_level=None):
    """Pontszám = középfrekvenciás részlet / √zaj.
    A beégett (telített) pixelek kimaradnak: ott nincs se részlet, se zaj, és torzítanák a mérést.
    A zajnak alsó korlátja van (8 bites kvantálás szintje), különben egy beégett, zajmentes
    csempén a pontszám „elszállna” (valódi napfelvételen +361% hamis javulást mutatott)."""
    if sat_level is not None:
        dil = cv2.dilate((img >= sat_level).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        mask = mask & ~dil
    if mask.sum() < 1000:
        return 0.0, 0.0, 0.0
    m = float(img[mask].mean())
    dog = cv2.GaussianBlur(img, (0, 0), 1.0) - cv2.GaussianBlur(img, (0, 0), 3.0)
    S = float(dog[mask].var()) / (m * m + 1e-12)
    hf = img - cv2.GaussianBlur(img, (0, 0), 0.7)
    N = float(1.4826 * np.median(np.abs(hf[mask]))) / (m + 1e-12)
    # alsó korlát csak a „közel nulla” zaj ellen (a beégett pixelek már ki vannak zárva); a valódi
    # zajnál jóval kisebb, hogy a Holdon ne torzítsa a választást (1/255 túl nagy volt: rossz boxot választott)
    N = max(N, (0.1 / 255.0) / max(m, 1e-3))
    return S / np.sqrt(N), S, N


def run_trials(tiles, ref_tile, mask_tile, gscore, half_boxes, percents, search, min_frames,
               struct_rel=0.05, min_corr=0.5, log=print, ap_erode=4, ap_cover=0.7):
    """tiles: lista (F) globálisan illesztett szürke csempe (-1 = érvénytelen); gscore: (F) globális élesség."""
    t0 = time.time()
    F = len(tiles)
    Hh, Ww = ref_tile.shape
    refb = cv2.GaussianBlur(ref_tile, (0, 0), 1.0)
    smask = geo.erode(mask_tile, 40)
    # beégés-küszöb csak akkor, ha a csempe ténylegesen közel van a telítéshez (Holdnál nincs hatása)
    rmax = float(ref_tile.max())
    sat_level = 0.985 * rmax if rmax >= 0.9 else None
    blurred = [cv2.GaussianBlur(np.where(t >= 0, t, 0).astype(np.float32), (0, 0), 1.0) for t in tiles]
    valids = [t >= 0 for t in tiles]
    results = []
    ntot = len(half_boxes) * len(percents)
    done = 0
    for hb in half_boxes:
        try:
            aps, _, step = apmod.make_grid(ref_tile, geo.erode(mask_tile, ap_erode), hb, 0, min_cover=ap_cover,
                                           struct_rel=struct_rel)
        except RuntimeError:
            continue
        if len(aps) < 6:
            continue
        sh = np.zeros((F, len(aps), 2), np.float32); ok = np.zeros((F, len(aps)), bool)
        q = np.zeros((F, len(aps)), np.float32)
        pts = aps[:, ::-1].astype(np.float64)
        for k in range(F):
            s, o, _ = apmod.measure(blurred[k], valids[k], refb, aps, hb, search, min_corr)
            sh[k], ok[k] = s, o
            q[k] = apmod.local_quality_at(np.where(valids[k], tiles[k], 0).astype(np.float32), pts, hb)
        raw_ok = float(ok.mean())
        sh, ok = apmod.smooth(sh, ok, apmod.Neighbors(aps, step))
        # ugyanaz a szabály, mint a teljes futásban: a kockák negyedében sem megbízható AP kimarad
        rel = ok.mean(0)
        keep = rel >= 0.25
        if keep.sum() < 3:
            keep = np.zeros_like(keep); keep[np.argsort(-rel)[:3]] = True
        aps, sh, ok, q = aps[keep], sh[:, keep], ok[:, keep], q[:, keep]
        field = apmod.Field(aps, (Hh, Ww), step)
        cg = CoarseGrid(Ww, Hh, field.f)
        for pct in percents:
            wsel = apmod.soft_select(q, ok, pct, min_frames)
            wg = apmod.soft_select_global(gscore, pct, min_frames)
            num = np.zeros((Hh, Ww), np.float32); den = np.zeros((Hh, Ww), np.float32)
            for k in range(F):
                if wsel[k].max() <= 0 and wg[k] <= 0:
                    continue
                mx, my = cg.up_map(cg.xc + field.shift(sh[k, :, 1], ok[k]), cg.yc + field.shift(sh[k, :, 0], ok[k]))
                out = cv2.remap(np.where(valids[k], tiles[k], 0).astype(np.float32), mx, my, cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                vv = cv2.remap(valids[k].astype(np.float32), mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.99
                wm = field.up(field.weight(wsel[k], wg[k])) * vv
                num += wm * out; den += wm
            st = num / np.maximum(den, 1e-6)
            sc, S, N = score(st, smask & (den > 1e-3), sat_level)
            done += 1
            results.append(dict(half_box=int(hb), percent=float(pct), score=float(sc), detail=float(S),
                                noise=float(N), aps=int(len(aps)), ap_success=raw_ok))
            log(f"  próba {done}/{ntot}: AP-box {2 * hb} px, legjobb {pct:g}% → pontszám {sc:.4f} "
                f"(részlet {S:.3e}, zaj {N:.4f})  PROGRESS {done}/{ntot}")
    if not results:
        return None, results
    best = max(results, key=lambda r: r["score"])
    top = best["score"]
    for r in results:
        r["relative"] = r["score"] / top
    log(f"  próbák ideje: {time.time() - t0:.0f} s")
    return best, results
