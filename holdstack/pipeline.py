"""A teljes folyamat:
elemzés -> globális illesztés -> flat -> referencia -> AP-rács -> torzítás-önkalibráció
-> (újra referencia + AP-rács torzítatlan térben) -> AP-mérés -> finomítás -> stack -> mentés.
"""
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
from . import distortion as distmod
from . import color as colormod


def src_is_color(src):
    for _i, raw in src.frames([0]):
        return raw.ndim == 3 and raw.shape[2] >= 3 and not (
            np.array_equal(raw[..., 0], raw[..., 1]) and np.array_equal(raw[..., 1], raw[..., 2]))
    return False
from .post import wavelet_sharpen, soft_clip, stretch, clipped_fraction, save16, save8


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
    distortion: bool = True        # szenzorhoz rögzített geometriai torzítás önkalibrációja
    dist_deg: int = 3
    dist_frames: int = 60          # ennyi (időben egyenletesen választott) képkockából kalibrál
    ca: bool = True                # kromatikus aberráció (laterális CA + légköri diszperzió) korrekciója
    ca_frames: int = 16            # ennyi képkockából méri a CA-mezőt
    defringe: float = 0.5          # színszegély-csillapítás erőssége az élesített képen (0 = ki)
    mineral: float = 0.0           # „mineral moon” színerősítés (0 = ki, tipikusan 3-6)
    sharpen: tuple = (2.0, 1.5, 1.2, 1.0)
    stretch: bool = False          # automatikus széthúzás (alapból KI: a fényesség = a bemenetével)
    knee: float = 0.85             # lágy csúcsfény-levágás kezdete az élesített képen
    workers: int = 0


def _log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


_T0 = [time.time()]


def _pmap(src, indices, fn, workers, depth=None, label=None):
    """Képkockák sorrendi olvasása, párhuzamos feldolgozás, sorrendtartó eredmény.
    Ha label meg van adva, kb. 10%-onként haladást ír ki ("PROGRESS label done/total")."""
    depth = depth or workers + 1
    total = src.n if indices is None else len(indices)
    step = max(1, total // 10)
    done = 0
    q = deque()

    def tick():
        nonlocal done
        done += 1
        if label and (done % step == 0 or done == total):
            print(f"[{time.time() - _T0[0]:7.1f}s]   {label}: {done}/{total}  PROGRESS {done}/{total}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        for i, raw in src.frames(indices):
            q.append(ex.submit(fn, i, raw))
            while len(q) > depth:
                r = q.popleft().result(); tick()
                yield r
        while q:
            r = q.popleft().result(); tick()
            yield r


def run(inp, outdir, cfg: Config = Config()):
    t0 = time.time()
    _T0[0] = t0
    os.makedirs(outdir, exist_ok=True)
    workers = cfg.workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    src = Source(inp)
    H, W = src.shape
    ds = cfg.ds
    _log(f"bemenet: {src.kind}, {src.n} kocka, {W}x{H}", t0)

    # ---- 1. elemzés: kicsinyített kép, Hold-maszk, globális élesség, telítettség -----------------
    def analyse(i, raw):
        f = to_float(raw)
        g = luminance(f)
        s = cv2.resize(g, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        m = geo.moon_mask(s)
        mf = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        sat = float((f.max(axis=2)[mf] >= 0.98).mean()) if mf.any() else 0.0
        return i, s.astype(np.float16), m, geo.sharpness(g, geo.erode(mf, 6 * ds)), sat

    smalls, masks, sharp, satin = {}, {}, {}, {}
    for i, s, m, q, st in _pmap(src, None, analyse, workers, label="elemzés"):
        smalls[i], masks[i], sharp[i], satin[i] = s, m, q, st
    n = len(sharp)
    ts = src.timestamps or list(range(n))
    names = src.names()[:n]
    _log(f"elemzés kész: {n} kocka; beégett pixelek a bemeneten (Hold): "
         f"medián {np.median(list(satin.values())) * 100:.3f}%, max {max(satin.values()) * 100:.3f}%", t0)

    order = sorted(sharp, key=lambda i: -sharp[i])
    ncand = n if n <= 60 else max(cfg.min_frames * 2, int(math.ceil(cfg.candidate_percent / 100 * n)))
    cand = sorted(order[:ncand])
    ref_i = order[0]

    # ---- 2. globális illesztés (homográfia, torzított térben) -------------------------------------
    ref_small = smalls[ref_i].astype(np.float32)
    ref_kp, ref_desc = geo.reference_features(ref_small, masks[ref_i])
    Hs_small, ecc, meth = {}, {}, {}

    def align(i):
        if i == ref_i:
            return i, np.eye(3), 1.0, "ref"
        h, c, m = geo.align_one(ref_small, masks[ref_i], ref_kp, ref_desc, smalls[i].astype(np.float32), masks[i])
        return i, h, c, m

    with ThreadPoolExecutor(workers) as ex:
        stepa = max(1, len(cand) // 10)
        for k, (i, h, c, m) in enumerate(ex.map(align, cand), 1):
            Hs_small[i], ecc[i], meth[i] = h, c, m
            if k % stepa == 0 or k == len(cand):
                print(f"[{time.time() - t0:7.1f}s]   globális illesztés: {k}/{len(cand)}  PROGRESS {k}/{len(cand)}",
                      flush=True)
    nweak = sum(1 for i in cand if not meth[i].endswith("+ecc") and i != ref_i)
    _log(f"globális illesztés kész: referencia={names[ref_i]}, {len(cand)} jelölt, "
         f"ECC medián={np.nanmedian([ecc[i] for i in cand]):.4f}, gyenge illesztés: {nweak}", t0)
    Hf = {i: geo.scale_h(Hs_small[i], ds) for i in cand}
    maps = {i: distmod.FrameMap(Hf[i]) for i in cand}

    # ---- 3. önkalibráló flat (fényesség-semleges normálással) ------------------------------------
    calib = dict(enabled=False, msg="kikapcsolva")
    if cfg.flat:
        calib = flatmod.calibrate(smalls, masks, Hs_small, cand, ref_i, deg=cfg.flat_deg)
    gflat = flatmod.flat_field(calib, W, H)
    inv_g = None if gflat is None else (1.0 / gflat)
    gains = {i: 0.0 for i in cand}
    if calib.get("enabled") and cfg.gain:
        gains.update({k: v for k, v in calib["gains"].items() if k in gains})
        mg = float(np.mean([gains[i] for i in cand]))
        gains = {i: g - mg for i, g in gains.items()}   # átlagos erősítés = 1 (nincs fényesség-eltolás)
    if calib.get("enabled"):
        _log(f"flat: vinyettálás {calib['g_min']:.3f}..{calib['g_max']:.3f} (átlag 1-re normálva), "
             f"maradék rms={calib['rms']:.4f}, erősítés szórás={np.std(list(gains.values())):.4f}", t0)
    else:
        _log(f"flat: nincs ({calib.get('msg')})", t0)
    del smalls

    ca_holder = [None]

    def corrected(i, raw, ca=True):
        f = to_float(raw)
        if inv_g is not None:
            f *= inv_g[..., None]
        if gains.get(i):
            f *= np.float32(math.exp(-gains[i]))
        if ca and ca_holder[0] is not None and f.ndim == 3:
            f = ca_holder[0].apply(f)
        return f

    # ---- 3b. kromatikus aberráció: szenzorhoz kötött R/B eltolási mező ---------------------------
    cainfo = dict(enabled=False, msg="kikapcsolva")
    if cfg.ca and src_is_color(src):
        k = min(cfg.ca_frames, len(cand))
        ca_set = [cand[j] for j in np.unique(np.round(np.linspace(0, len(cand) - 1, k)).astype(int))]
        samples = {colormod.R: [], colormod.B: []}

        def ca_one(i, raw):
            f = corrected(i, raw, ca=False)
            m = cv2.resize(masks[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            return colormod.measure(f, geo.erode(m, 30))

        for res in _pmap(src, ca_set, ca_one, workers, label="színhiba-mérés"):
            for c in samples:
                samples[c].append(res[c])
        samples = {c: np.concatenate(v) if v else np.zeros((0, 4)) for c, v in samples.items()}
        ca_field, cainfo = colormod.CAField.fit(samples, W, H, deg=2)
        if ca_field is not None:
            ca_holder[0] = ca_field
            _log(f"kromatikus aberráció: R eltolás átlag {cainfo['R']['mean_shift_px']:.2f} px "
                 f"(max {cainfo['R']['max_shift_px']:.2f}), B átlag {cainfo['B']['mean_shift_px']:.2f} px "
                 f"(max {cainfo['B']['max_shift_px']:.2f}); minden képkockán korrigálva", t0)
        else:
            _log(f"kromatikus aberráció: nincs korrekció ({cainfo.get('msg')})", t0)

    gy, gx = np.mgrid[0:H, 0:W].astype(np.float32)

    def warp_gray(i, raw):
        g = luminance(corrected(i, raw))
        return i, g, maps[i].warp(g, W, H, gx, gy, cv2.INTER_LINEAR, -1.0)

    nref = min(len(cand), max(cfg.ref_min, int(math.ceil(cfg.ref_percent / 100 * len(cand)))))
    ref_set = [i for i in order if i in Hf][:nref]
    ref_mask_d = cv2.resize(masks[ref_i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    def build_reference():
        acc = np.zeros((H, W), np.float64); cnt = np.zeros((H, W), np.float64)
        for i, g, w in _pmap(src, ref_set, warp_gray, workers, label="referencia"):
            v = w >= 0
            acc[v] += w[v]; cnt[v] += 1
        ref = (acc / np.maximum(cnt, 1)).astype(np.float32)
        rm = maps[ref_i].warp(ref_mask_d, W, H, gx, gy, cv2.INTER_NEAREST, 0.0) > 0.5
        rm &= cnt >= max(1, nref // 2)
        aps, struct, step = apmod.make_grid(ref, geo.erode(rm, 4), cfg.half_box, cfg.step, struct_rel=cfg.struct_rel)
        return ref, rm, aps, step

    def measure_all(reference, aps, frames, label="AP-mérés"):
        refb = cv2.GaussianBlur(reference, (0, 0), 1.0)
        pos = {i: k for k, i in enumerate(frames)}
        F = len(frames)
        sh = np.zeros((F, len(aps), 2), np.float32); ok = np.zeros((F, len(aps)), bool)
        q = np.zeros((F, len(aps)), np.float32)

        def one(i, raw):
            _, g, w = warp_gray(i, raw)
            valid = w >= 0
            wb = cv2.GaussianBlur(np.where(valid, w, 0).astype(np.float32), (0, 0), 1.0)
            s, o, _c = apmod.measure(wb, valid, refb, aps, cfg.half_box, cfg.search, cfg.min_corr)
            pts = maps[i].points(aps[:, ::-1].astype(np.float64))
            return i, s, o, apmod.local_quality_at(g, pts, cfg.half_box)

        for i, s, o, qq in _pmap(src, frames, one, workers, label=label):
            k = pos[i]
            sh[k], ok[k], q[k] = s, o, qq
        return sh, ok, q

    # ---- 4. kezdeti referencia és AP-rács (torzított térben) -----------------------------------
    ref0, ref_mask, aps, step = build_reference()
    _log(f"kezdeti referencia: {nref} kocka átlaga; AP-rács: {len(aps)} pont", t0)

    # ---- 5. torzítás-önkalibráció ------------------------------------------------------------------
    dinfo = dict(enabled=False, msg="kikapcsolva")
    dist = None
    if cfg.distortion and len(cand) >= 3:
        if len(cand) > cfg.dist_frames:
            sel = np.unique(np.round(np.linspace(0, len(cand) - 1, cfg.dist_frames)).astype(int))
            dframes = [cand[k] for k in sel]
        else:
            dframes = list(cand)
        sh0, ok0, _ = measure_all(ref0, aps, dframes, label="torzítás-mérés")
        nb0 = apmod.Neighbors(aps, step)
        sh0, ok0 = apmod.smooth(sh0, ok0, nb0)
        corr = []
        p = aps[:, ::-1].astype(np.float64)  # (x, y)
        for k, i in enumerate(dframes):
            o = ok0[k]
            xr = p[o]
            xk = cv2.perspectiveTransform((p[o] + sh0[k][o][:, ::-1])[None], Hf[i])[0]
            corr.append((xr, xk))
        dist, dinfo = distmod.calibrate(corr, W, H, deg=cfg.dist_deg)
        if dist is not None:
            ys, xs = np.nonzero(geo.erode(ref_mask, 20))
            sel = np.random.default_rng(1).choice(len(xs), min(4000, len(xs)), replace=False)
            pts_ref = np.stack([xs[sel], ys[sel]], 1).astype(np.float64)
            maps = {i: distmod.FrameMap(distmod.frame_homography(dist, Hf[i], pts_ref), dist) for i in cand}
            dinfo["max_shift_moon_px"] = distmod.max_shift_in(dist, ref_mask)
            cv2.imwrite(os.path.join(outdir, "_torzitas_mezo.png"), distmod.draw_field(dist, W, H, ref_mask))
            _log(f"torzítás: {dinfo['frames']} kocka, {dinfo['points']} pontpár; maradék rms "
                 f"{dinfo['rms_before']:.2f} -> {dinfo['rms_after']:.2f} px (medián {dinfo['median_before']:.2f} -> "
                 f"{dinfo['median_after']:.2f}); max korrekció a Hold területén {dinfo['max_shift_moon_px']:.1f} px", t0)
            ref0, ref_mask, aps, step = build_reference()
            _log(f"referencia torzítatlan térben újraépítve; AP-rács: {len(aps)} pont", t0)
        else:
            _log(f"torzítás: nem alkalmazva ({dinfo.get('msg')}"
                 + (f", rms {dinfo['rms_before']:.2f} -> {dinfo['rms_after']:.2f} px" if 'rms_before' in dinfo else "")
                 + ")", t0)

    nb = apmod.Neighbors(aps, step)
    field = apmod.Field(aps, (H, W), step)
    cand_pos = {i: k for k, i in enumerate(cand)}
    _log(f"AP-rács: {len(aps)} pont, box={2 * cfg.half_box}px, lépés={step}px", t0)

    # ---- 6. stack (warp-mező) -----------------------------------------------------------------------
    def stack(sh, ok, wap, wglob, color, interp=cv2.INTER_CUBIC, label="stack"):
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
            mx, my = maps[i].grid(gx + dx, gy + dy)
            out = cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_REFLECT)
            wm = wm * ((mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1))
            return wm.astype(np.float32), out

        for wm, out in _pmap(src, need, one, max(1, min(workers, 4)), depth=2, label=label):
            if out.ndim == 2:
                out = out[..., None]
            num += wm[..., None] * out
            den += wm
        res = num / np.maximum(den, 1e-6)[..., None]
        return (res if color else res[..., 0]), den, len(need)

    def measure_smooth(reference):
        sh, ok, q = measure_all(reference, aps, cand)
        raw_ok = ok.copy()
        sh, ok = apmod.smooth(sh, ok, nb)
        return sh, ok, q, raw_ok

    sh, ok, q, raw_ok = measure_smooth(ref0)
    _log(f"AP-mérés: sikeres {raw_ok.mean() * 100:.1f}%, simítás után megbízható {ok.mean() * 100:.1f}%", t0)

    gscore = np.array([sharp[i] for i in cand])
    if cfg.refine:
        wref = apmod.soft_select(q, ok, cfg.ref_percent, cfg.ref_min)
        ref1, _, nused = stack(sh, ok, wref, apmod.soft_select_global(gscore, cfg.ref_percent, cfg.ref_min),
                               color=False, interp=cv2.INTER_LINEAR, label="finomított referencia")
        _log(f"finomított referencia kész ({nused} kocka)", t0)
        sh, ok, q, raw_ok = measure_smooth(ref1)
        _log(f"újramérés: sikeres {raw_ok.mean() * 100:.1f}%, megbízható {ok.mean() * 100:.1f}%", t0)

    wsel = apmod.soft_select(q, ok, cfg.stack_percent, cfg.min_frames)
    wglob = apmod.soft_select_global(gscore, cfg.stack_percent, cfg.min_frames)
    result, den, nused = stack(sh, ok, wsel, wglob, color=True)
    _log(f"stack kész: {nused} kocka járult hozzá", t0)

    # ---- 7. kivágás, fényesség, mentés, jelentés -------------------------------------------------
    ys, xs = np.nonzero(ref_mask)
    pad = 60
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    # maradék CA + légköri diszperzió a kész stacken (kimeneti koordinátákban, lineáris mező)
    ca2 = dict(enabled=False, msg="kikapcsolva")
    if cfg.ca and result.ndim == 3:
        s2 = colormod.measure(np.clip(result, 0, None), geo.erode(ref_mask, 30), step=56)
        f2, ca2 = colormod.CAField.fit(s2, W, H, deg=1, min_pts=30)
        if f2 is not None and max(ca2["R"]["max_shift_px"], ca2["B"]["max_shift_px"]) > 0.05:
            result = f2.apply(result)
            _log(f"maradék színhiba a stacken: R {ca2['R']['mean_shift_px']:.2f} px, "
                 f"B {ca2['B']['mean_shift_px']:.2f} px — korrigálva", t0)
    crop = np.clip(result[y0:y1, x0:x1], 0, None)
    cm = ref_mask[y0:y1, x0:x1]
    # A fényesség a bemenetével azonos (súlyozott átlag); széthúzás csak kérésre, fej-térrel.
    lin = stretch(crop, cm) if cfg.stretch else crop
    lin = np.clip(lin, 0, 1)
    sharp_img = soft_clip(np.maximum(wavelet_sharpen(colormod.defringe(lin, cfg.defringe), cfg.sharpen), 0),
                          cfg.knee)
    bright = dict(
        input_saturated_median=float(np.median(list(satin.values()))),
        input_saturated_max=float(max(satin.values())),
        stack_saturated=clipped_fraction(lin, cm), sharpened_saturated=clipped_fraction(sharp_img, cm),
        stack_mean=float(lin[cm].mean()))
    base = os.path.splitext(os.path.basename(os.path.normpath(inp)))[0]
    save16(os.path.join(outdir, f"{base}_stack.tif"), lin)
    save16(os.path.join(outdir, f"{base}_stack_sharpened.tif"), sharp_img)
    save8(os.path.join(outdir, f"{base}_preview.png"), sharp_img)
    if cfg.mineral > 0 and lin.ndim == 3:
        mineral = colormod.mineral_moon(colormod.defringe(lin, max(cfg.defringe, 0.7)), cfg.mineral, cm,
                                        luminance=sharp_img)
        save16(os.path.join(outdir, f"{base}_mineral.tif"), mineral)
        save8(os.path.join(outdir, f"{base}_mineral.png"), mineral)
        _log(f"ásványi hold (mineral moon) kép elkészült, erősség {cfg.mineral:g}", t0)
    _log(f"fényesség: beégett pixel a Holdon — bemenet max {bright['input_saturated_max'] * 100:.3f}%, "
         f"stack {bright['stack_saturated'] * 100:.3f}%, élesített {bright['sharpened_saturated'] * 100:.3f}%", t0)

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
    if dist is not None:
        os.replace(os.path.join(outdir, "_torzitas_mezo.png"), os.path.join(outdir, f"{base}_torzitas.png"))

    shmag = np.hypot(sh[..., 0], sh[..., 1])[ok]
    report = dict(
        config=asdict(cfg), input=inp, kind=src.kind, frames=n, size=[W, H], reference=names[ref_i],
        candidates=len(cand), contributing=nused, aps=len(aps), ap_step=step,
        ap_success=float(raw_ok.mean()), ap_reliable=float(ok.mean()),
        local_shift_px=dict(median=float(np.median(shmag)) if shmag.size else None,
                            p95=float(np.percentile(shmag, 95)) if shmag.size else None),
        flat={k: v for k, v in calib.items() if k != "gains"}, distortion=dinfo, brightness=bright,
        chromatic_aberration=dict(per_frame=cainfo, residual_on_stack=ca2),
        frames_detail=[dict(name=names[i], t=float(ts[i]) if i < len(ts) else None, sharp=sharp[i],
                            saturated=satin[i], candidate=i in cand_pos, ecc=ecc.get(i), method=meth.get(i),
                            gain=gains.get(i), global_weight=float(wglob[cand_pos[i]]) if i in cand_pos else 0.0,
                            ap_weight_mean=float(wsel[cand_pos[i]].mean()) if i in cand_pos else 0.0)
                       for i in range(n)],
        crop=[int(x0), int(y0), int(x1), int(y1)], seconds=time.time() - t0)
    with open(os.path.join(outdir, f"{base}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=float)
    _log(f"kész: {outdir}", t0)
    return report
