"""A teljes folyamat: elemzés -> globális illesztés -> flat -> referencia -> AP-mérés -> (finomítás) -> stack."""
import os
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

import numpy as np
import cv2

from .source import Source, to_float, luminance
from . import geometry as geo
from . import flat as flatmod
from . import ap as apmod
from .post import wavelet_sharpen, save16, save8


@dataclass
class Config:
    ds: int = 4                    # elemzési kicsinyítés (globális illesztés, flat)
    candidate_percent: float = 50  # videónál a globálisan legjobb ennyi % megy tovább (<= 60 kockánál mind)
    stack_percent: float = 20      # AP-nként a legjobb ennyi % kerül a stackbe
    min_frames: int = 5            # de legalább ennyi képkocka
    ref_percent: float = 5         # referenciához használt legjobb %
    ref_min: int = 3
    half_box: int = 32             # AP-box félszélessége (px)
    step: int = 0                  # AP-rácsosztás (0 = 1.5 * half_box)
    search: int = 16               # lokális keresési sugár (px)
    min_corr: float = 0.5
    struct_rel: float = 0.05
    refine: bool = True            # második mérés egy élesebb, AP-alapú referenciához
    flat: bool = True              # önkalibráló vinyettálás-korrekció
    flat_deg: int = 3
    gain: bool = True              # képkockánkénti erősítés-kiegyenlítés
    sharpen: tuple = (2.0, 1.5, 1.2, 1.0)
    workers: int = 0


def _log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _pmap(src, indices, fn, workers, depth=None):
    """Képkockák sorrendi olvasása, párhuzamos feldolgozás, sorrendtartó eredmény."""
    depth = depth or workers + 1
    q = deque()
    with ThreadPoolExecutor(workers) as ex:
        for i, raw in src.frames(indices):
            q.append(ex.submit(fn, i, raw))
            while len(q) > depth:
                yield q.popleft().result()
        while q:
            yield q.popleft().result()


def run(inp, outdir, cfg: Config = Config()):
    t0 = time.time()
    os.makedirs(outdir, exist_ok=True)
    workers = cfg.workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    src = Source(inp)
    H, W = src.shape
    ds = cfg.ds
    _log(f"bemenet: {src.kind}, {src.n} kocka, {W}x{H}", t0)

    # ---- 1. elemzés: kicsinyített kép, Hold-maszk, globális élesség -------------------------------
    def analyse(i, raw):
        f = to_float(raw)
        g = luminance(f)
        s = cv2.resize(g, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        m = geo.moon_mask(s)
        mf = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        return i, s.astype(np.float16), m, geo.sharpness(g, geo.erode(mf, 6 * ds))

    smalls, masks, sharp = {}, {}, {}
    for i, s, m, q in _pmap(src, None, analyse, workers):
        smalls[i], masks[i], sharp[i] = s, m, q
    n = len(sharp)
    ts = src.timestamps or list(range(n))
    names = src.names()[:n]
    _log(f"elemzés kész: {n} kocka", t0)

    order = sorted(sharp, key=lambda i: -sharp[i])
    ncand = n if n <= 60 else max(cfg.min_frames * 2, int(math.ceil(cfg.candidate_percent / 100 * n)))
    cand = sorted(order[:ncand])
    ref_i = order[0]

    # ---- 2. globális illesztés (homográfia) --------------------------------------------------------
    ref_small = smalls[ref_i].astype(np.float32)
    ref_kp, ref_desc = geo.reference_features(ref_small, masks[ref_i])
    Hs_small, ecc, meth = {}, {}, {}

    def align(i):
        if i == ref_i:
            return i, np.eye(3), 1.0, "ref"
        h, c, m = geo.align_one(ref_small, masks[ref_i], ref_kp, ref_desc, smalls[i].astype(np.float32), masks[i])
        return i, h, c, m

    with ThreadPoolExecutor(workers) as ex:
        for i, h, c, m in ex.map(align, cand):
            Hs_small[i], ecc[i], meth[i] = h, c, m
    bad = [i for i in cand if not meth[i].endswith("+ecc") and i != ref_i]
    _log(f"globális illesztés kész: referencia={names[ref_i]}, {len(cand)} jelölt, "
         f"ECC medián={np.nanmedian([ecc[i] for i in cand]):.4f}, gyenge illesztés: {len(bad)}", t0)
    Hf = {i: geo.scale_h(Hs_small[i], ds) for i in cand}

    # ---- 3. önkalibráló flat -----------------------------------------------------------------------
    calib = dict(enabled=False, msg="kikapcsolva")
    if cfg.flat:
        calib = flatmod.calibrate(smalls, masks, Hs_small, cand, ref_i, deg=cfg.flat_deg)
    gflat = flatmod.flat_field(calib, W, H)
    inv_g = None if gflat is None else (1.0 / gflat)
    gains = {i: 0.0 for i in cand}
    if calib.get("enabled") and cfg.gain:
        gains.update({k: v for k, v in calib["gains"].items()})
    if calib.get("enabled"):
        _log(f"flat: vinyettálás {calib['g_min']:.3f}..{calib['g_max']:.3f}, maradék rms={calib['rms']:.4f}, "
             f"erősítés szórás={np.std(list(gains.values())):.4f}", t0)
    else:
        _log(f"flat: nincs ({calib.get('msg')})", t0)
    del smalls

    def corrected(i, raw):
        f = to_float(raw)
        if inv_g is not None:
            f *= inv_g[..., None]
        if gains.get(i):
            f *= np.float32(math.exp(-gains[i]))
        return f

    # ---- 4. kezdeti referencia: a legjobb képkockák átlaga csak homográfiával ------------------
    nref = min(len(cand), max(cfg.ref_min, int(math.ceil(cfg.ref_percent / 100 * len(cand)))))
    ref_set = [i for i in order if i in Hf][:nref]

    def warp_gray(i, raw):
        g = luminance(corrected(i, raw))
        w = cv2.warpPerspective(g, Hf[i], (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        return i, g, w

    acc = np.zeros((H, W), np.float64); cnt = np.zeros((H, W), np.float64)
    for i, g, w in _pmap(src, ref_set, warp_gray, workers):
        v = w >= 0
        acc[v] += w[v]; cnt[v] += 1
    ref0 = (acc / np.maximum(cnt, 1)).astype(np.float32)
    ref_mask = cv2.resize(masks[ref_i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    ref_mask &= cnt >= max(1, nref // 2)
    _log(f"kezdeti referencia: {nref} kocka átlaga", t0)

    # ---- 5. AP-rács ---------------------------------------------------------------------------------
    aps, struct, step = apmod.make_grid(ref0, geo.erode(ref_mask, 4), cfg.half_box, cfg.step,
                                        struct_rel=cfg.struct_rel)
    nb = apmod.Neighbors(aps, step)
    field = apmod.Field(aps, (H, W), step)
    _log(f"AP-rács: {len(aps)} pont, box={2 * cfg.half_box}px, lépés={step}px", t0)
    cand_pos = {i: k for k, i in enumerate(cand)}

    # ---- 6. lokális mérés ---------------------------------------------------------------------------
    def measure_all(reference):
        refb = cv2.GaussianBlur(reference, (0, 0), 1.0)
        F = len(cand)
        sh = np.zeros((F, len(aps), 2), np.float32); ok = np.zeros((F, len(aps)), bool)
        q = np.zeros((F, len(aps)), np.float32); cr = np.zeros((F, len(aps)), np.float32)

        def one(i, raw):
            _, g, w = warp_gray(i, raw)
            valid = w >= 0
            wb = cv2.GaussianBlur(np.where(valid, w, 0).astype(np.float32), (0, 0), 1.0)
            s, o, c = apmod.measure(wb, valid, refb, aps, cfg.half_box, cfg.search, cfg.min_corr)
            return i, s, o, c, apmod.local_quality(g, Hf[i], aps, cfg.half_box)

        for i, s, o, c, qq in _pmap(src, cand, one, workers):
            k = cand_pos[i]
            sh[k], ok[k], cr[k], q[k] = s, o, c, qq
        raw_ok = ok.copy()
        sh, ok = apmod.smooth(sh, ok, nb)
        return sh, ok, q, raw_ok, cr

    # ---- 7. stack (warp-mező) -----------------------------------------------------------------------
    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)

    def stack(sh, ok, wap, wglob, color, interp=cv2.INTER_CUBIC):
        need = [i for i in cand if wap[cand_pos[i]].max() > 0 or wglob[cand_pos[i]] > 0]
        ch = 3 if color else 1
        num = np.zeros((H, W, ch), np.float32); den = np.zeros((H, W), np.float32)

        def one(i, raw):
            k = cand_pos[i]
            f = corrected(i, raw)
            img = f if color else luminance(f)
            dy = field.up(field.shift(sh[k, :, 0], ok[k]))
            dx = field.up(field.shift(sh[k, :, 1], ok[k]))
            wm = field.up(field.weight(wap[k], wglob[k]))
            xs = gx + dx; ys = gy + dy
            h = Hf[i]
            z = h[2, 0] * xs + h[2, 1] * ys + h[2, 2]
            mx = ((h[0, 0] * xs + h[0, 1] * ys + h[0, 2]) / z).astype(np.float32)
            my = ((h[1, 0] * xs + h[1, 1] * ys + h[1, 2]) / z).astype(np.float32)
            out = cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REFLECT)
            wm = wm * ((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1))
            return wm.astype(np.float32), out

        for wm, out in _pmap(src, need, one, max(1, min(workers, 4)), depth=2):
            if out.ndim == 2:
                out = out[..., None]
            num += wm[..., None] * out
            den += wm
        res = num / np.maximum(den, 1e-6)[..., None]
        return (res if color else res[..., 0]), den, len(need)

    sh, ok, q, raw_ok, cr = measure_all(ref0)
    _log(f"AP-mérés: sikeres {raw_ok.mean() * 100:.1f}%, simítás után megbízható {ok.mean() * 100:.1f}%", t0)

    gscore = np.array([sharp[i] for i in cand])
    if cfg.refine:
        wref = apmod.soft_select(q, ok, cfg.ref_percent, cfg.ref_min)
        ref1, _, nused = stack(sh, ok, wref, apmod.soft_select_global(gscore, cfg.ref_percent, cfg.ref_min),
                               color=False, interp=cv2.INTER_LINEAR)
        _log(f"finomított referencia kész ({nused} kocka)", t0)
        sh, ok, q, raw_ok, cr = measure_all(ref1)
        _log(f"újramérés: sikeres {raw_ok.mean() * 100:.1f}%, megbízható {ok.mean() * 100:.1f}%", t0)

    wsel = apmod.soft_select(q, ok, cfg.stack_percent, cfg.min_frames)
    wglob = apmod.soft_select_global(gscore, cfg.stack_percent, cfg.min_frames)
    result, den, nused = stack(sh, ok, wsel, wglob, color=True)
    _log(f"stack kész: {nused} kocka járult hozzá", t0)

    # ---- 8. kivágás, mentés, jelentés --------------------------------------------------------------
    m = ref_mask
    ys, xs = np.nonzero(m)
    pad = 60
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    crop = result[y0:y1, x0:x1]
    lo = np.percentile(crop, 0.05); hi = np.percentile(crop[m[y0:y1, x0:x1]], 99.95)
    lin = np.clip((crop - lo) / max(hi - lo, 1e-6), 0, 1)
    sharp_img = np.clip(wavelet_sharpen(lin, cfg.sharpen), 0, 1)
    base = os.path.splitext(os.path.basename(os.path.normpath(inp)))[0]
    save16(os.path.join(outdir, f"{base}_stack.tif"), lin)
    save16(os.path.join(outdir, f"{base}_stack_sharpened.tif"), sharp_img)
    save8(os.path.join(outdir, f"{base}_preview.png"), sharp_img)

    # AP-térkép
    vis = cv2.cvtColor(np.clip(ref0 * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rel = ok.mean(0)
    for (y, x), r in zip(aps, rel):
        col = (0, int(255 * r), int(255 * (1 - r)))
        cv2.rectangle(vis, (int(x - cfg.half_box), int(y - cfg.half_box)),
                      (int(x + cfg.half_box), int(y + cfg.half_box)), col, 2)
    cv2.imwrite(os.path.join(outdir, f"{base}_aps.jpg"), cv2.resize(vis, (W // 2, H // 2)),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    if gflat is not None:
        save8(os.path.join(outdir, f"{base}_flat.png"), gflat / gflat.max())

    shmag = np.hypot(sh[..., 0], sh[..., 1])[ok]
    report = dict(
        config=asdict(cfg), input=inp, kind=src.kind, frames=n, size=[W, H], reference=names[ref_i],
        candidates=len(cand), contributing=nused, aps=len(aps), ap_step=step,
        ap_success=float(raw_ok.mean()), ap_reliable=float(ok.mean()),
        local_shift_px=dict(median=float(np.median(shmag)) if shmag.size else None,
                            p95=float(np.percentile(shmag, 95)) if shmag.size else None),
        flat={k: v for k, v in calib.items() if k != "gains"},
        frames_detail=[dict(name=names[i], t=float(ts[i]) if i < len(ts) else None, sharp=sharp[i],
                            candidate=i in cand_pos, ecc=ecc.get(i), method=meth.get(i),
                            gain=gains.get(i), global_weight=float(wglob[cand_pos[i]]) if i in cand_pos else 0.0,
                            ap_weight_mean=float(wsel[cand_pos[i]].mean()) if i in cand_pos else 0.0)
                       for i in range(n)],
        crop=[int(x0), int(y0), int(x1), int(y1)], seconds=time.time() - t0)
    with open(os.path.join(outdir, f"{base}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=float)
    _log(f"kész: {outdir}", t0)
    return report
