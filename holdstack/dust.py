"""Szenzorhoz rögzített porszemek (és más kis sötét hibák) feltérképezése és eltávolítása.

Elv: a nem követő Dobson miatt a célpont (Nap/Hold) vándorol a szenzoron, a porszem viszont a
szenzorhoz (vagy az okulárhoz/telefonhoz) rögzített. Minden mintakockára kiszámoljuk
    arány_k(x) = képkocka_k(x) / referencia(a célpont ugyanazon pontja)
a SZENZOR koordinátáiban. A valódi napfolt/kráter a célponttal együtt mozog, így a referencia
is tartalmazza → az arányban nem jelenik meg; a porszem viszont minden kockában ugyanott
okoz sötétedést → a kockák mediánjában megmarad. A medián nagyfrekvenciás része (a lassú
vinyettálást a flat kezeli) adja a pormaszkot; csak a zajszintnél szignifikánsan sötétebb,
kis kiterjedésű helyeken korrigálunk, máshol a maszk pontosan 1 (nem visz be zajt).
"""
import numpy as np
import cv2

from .source import to_float, luminance
from .distortion import CoarseGrid


def estimate(src, frames, corrected, maps, ref, ref_mask, W, H, masks_small, ds, log,
             min_drift_px=15.0, k_sigma=4.0, max_depth=0.6):
    """Visszaad: (pormaszk teljes szenzorméretben [float32, 1 = nincs por] vagy None, info dict)."""
    # a célpont szenzorbeli helyének szórása (elég mozgás kell, különben a por nem különíthető el)
    cents = []
    for i in frames:
        ys, xs = np.nonzero(masks_small[i])
        if len(xs):
            cents.append((xs.mean() * ds, ys.mean() * ds))
    cents = np.array(cents)
    drift = float(np.hypot(*cents.std(0))) if len(cents) > 1 else 0.0
    if len(frames) < 8 or drift < min_drift_px:
        return None, dict(enabled=False, drift_px=drift,
                          msg=f"kevés kocka vagy kevés mozgás a szenzoron (szórás {drift:.0f} px)")
    # a célpont által bejárt szenzorterület (befoglaló téglalap)
    union = np.zeros_like(masks_small[frames[0]], bool)
    for i in frames:
        union |= masks_small[i]
    ys, xs = np.nonzero(union)
    y0, y1 = max(0, ys.min() * ds - 8), min(H, (ys.max() + 1) * ds + 8)
    x0, x1 = max(0, xs.min() * ds - 8), min(W, (xs.max() + 1) * ds + 8)
    bh, bw = y1 - y0, x1 - x0
    cg = CoarseGrid(bw, bh, 8)
    refv = float(np.percentile(ref[ref_mask], 50)) if ref_mask.any() else float(ref.max())
    stack = np.full((len(frames), bh, bw), np.nan, np.float16)
    ok_count = 0
    for k, (i, raw) in enumerate(src.frames(frames)):
        g = luminance(corrected(i, raw))[y0:y1, x0:x1]
        ox, oy = maps[i].inv_map_xy(cg.xc + x0, cg.yc + y0)
        mx, my = cg.up_map(ox, oy)
        model = cv2.remap(ref, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mm = cv2.remap(ref_mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0) > 0.99
        # csak a korong biztos belsején (a perem közelében a modell-élesség eltérése torzítana)
        mm = cv2.erode(mm.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & (model > 0.15 * refv)
        if mm.sum() < 1000:
            continue
        r = np.full(g.shape, np.nan, np.float32)
        r[mm] = g[mm] / model[mm]
        r /= np.nanmedian(r[mm])
        stack[k] = r.astype(np.float16)
        ok_count += 1
        log(f"  pormaszk – mintakockák: {k + 1}/{len(frames)}  PROGRESS {k + 1}/{len(frames)}")
    if ok_count < 8:
        return None, dict(enabled=False, drift_px=drift, msg="kevés használható mintakocka")
    # medián soronként (memóriakímélő)
    med = np.full((bh, bw), np.nan, np.float32)
    cnt = np.zeros((bh, bw), np.int16)
    for r0 in range(0, bh, 128):
        blk = stack[:, r0:r0 + 128].astype(np.float32)
        med[r0:r0 + 128] = np.nanmedian(blk, axis=0) if np.isfinite(blk).any() else np.nan
        cnt[r0:r0 + 128] = np.isfinite(blk).sum(0)
    good = np.isfinite(med) & (cnt >= max(6, ok_count // 3))
    medf = np.where(good, med, 1.0).astype(np.float32)
    # nagyfrekvenciás rész: a lassú változást (vinyettálás-maradék, modell-elmosódás) leválasztjuk
    w = good.astype(np.float32)
    low = cv2.GaussianBlur(medf * w, (0, 0), 12) / np.maximum(cv2.GaussianBlur(w, (0, 0), 12), 1e-3)
    hp = np.where(good, medf / np.maximum(low, 1e-3), 1.0)
    sig = 1.4826 * float(np.median(np.abs(hp[good] - 1))) + 1e-4
    dark = good & (hp < 1 - k_sigma * sig)
    # csak kis, kompakt foltok (porszem); nagy kiterjedésű eltérés nem por
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    keep = np.zeros_like(dark)
    specks = []
    min_depth = max(0.03, 6 * sig)  # csak a biztos porszem: a zajszintű találat korrekciója zajt vinne be
    for j in range(1, n):
        a = st[j, cv2.CC_STAT_AREA]
        if not (6 <= a <= 2500):
            continue
        sel = lab == j
        dep = float(1 - hp[sel].min())
        if dep < min_depth:
            continue
        keep |= sel
        specks.append(dict(x=float(st[j, 0] + st[j, 2] / 2 + x0), y=float(st[j, 1] + st[j, 3] / 2 + y0),
                           area=int(a), depth=dep))
    if not specks:
        return None, dict(enabled=False, drift_px=drift, noise=sig, msg="nem találtam porszemet")
    # a porszem teljes profilja (a küszöb alatti szélekkel együtt): kicsit tágított terület
    zone = cv2.dilate(keep.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    zone = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), 1.5)
    corr = 1 + (np.clip(hp, 1 - max_depth, 1.0) - 1) * np.clip(zone, 0, 1)
    dmap = np.ones((H, W), np.float32)
    dmap[y0:y1, x0:x1] = corr
    return dmap, dict(enabled=True, drift_px=drift, noise=sig, frames=ok_count, specks=len(specks),
                      speck_list=sorted(specks, key=lambda d: -d["depth"])[:30],
                      max_depth=float(max(s["depth"] for s in specks)), msg="ok")
