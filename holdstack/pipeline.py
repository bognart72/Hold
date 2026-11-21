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
from dataclasses import dataclass, asdict, replace

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
from .post import wavelet_sharpen, soft_clip, stretch, clipped_fraction, save16, save8, sun_enhance


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
    dist_frames: int = 60          # ennyi (időben egyenletesen választott) képkockából kalibrál (30-cal mérhetően rosszabb)
    ca: bool = True                # kromatikus aberráció (laterális CA + légköri diszperzió) korrekciója
    ca_frames: int = 16            # ennyi képkockából méri a CA-mezőt
    defringe: float = 0.5          # színszegély-csillapítás erőssége az élesített képen (0 = ki)
    mineral: float = 0.0           # „mineral moon” színerősítés (0 = ki, tipikusan 3-6)
    sharpen: tuple = (2.0, 1.5, 1.2, 1.0)
    target: str = "moon"           # "moon" (Hold) vagy "sun" (Nap, fehér fényű napszűrővel)
    dust: object = None            # szenzorhoz rögzített porszemek eltávolítása (None = Napnál be, Holdnál ki)
    sun_enhance: bool = True       # Nap: szélsötétedés-kiegyenlített, kiemelt változat is készül
    trial: bool = False            # próbafuttatásos kalibráció (AP-box × arány) a teljes feldolgozás előtt
    trial_only: bool = False       # csak kalibráció: a javaslatot elmenti, stack nem készül
    trial_frames: int = 80         # legfeljebb ennyi képkockán fut a kalibráció
    trial_tile: int = 1024         # a kalibrációs csempe mérete (px)
    processes: bool = True         # AP-mérés külön folyamatokban (GIL nélkül, valódi párhuzamosság)
    stretch: bool = False          # automatikus széthúzás (alapból KI: a fényesség = a bemenetével)
    knee: float = 0.85             # lágy csúcsfény-levágás kezdete az élesített képen
    workers: int = 0


def _r4(x):
    return int(4 * round(x / 4.0))


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


def _ppmap(src, indices, ctx, nproc, label=None):
    """Mint a _pmap, de külön FOLYAMATOKBAN (holdstack.worker.measure_frame): a GIL-kötött
    AP-mérés így valóban párhuzamos. A képkockákat a fő folyamat olvassa, sorrendben adja vissza."""
    from concurrent.futures import ProcessPoolExecutor
    from . import worker
    total = len(indices)
    step = max(1, total // 10)
    done = 0
    q = deque()
    depth = nproc * 2
    with ProcessPoolExecutor(nproc, initializer=worker.init, initargs=(ctx,)) as ex:
        def tick():
            nonlocal done
            done += 1
            if label and (done % step == 0 or done == total):
                print(f"[{time.time() - _T0[0]:7.1f}s]   {label}: {done}/{total}  PROGRESS {done}/{total}", flush=True)
        for i, raw in src.frames(indices):
            q.append(ex.submit(worker.measure_frame, i, raw))
            while len(q) > depth:
                r = q.popleft().result(); tick()
                yield r
        while q:
            r = q.popleft().result(); tick()
            yield r


def run(inp, outdir, cfg: Config = Config()):
    t0 = time.time()
    _T0[0] = t0
    cfg = replace(cfg)  # a kalibráció módosíthatja; a hívó példánya érintetlen marad
    os.makedirs(outdir, exist_ok=True)
    workers = cfg.workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    nproc = cfg.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    src = Source(inp)
    H, W = src.shape
    ds = cfg.ds
    _log(f"bemenet: {src.kind}, {src.n} kocka, {W}x{H}", t0)
    sun = cfg.target == "sun"
    use_dust = cfg.dust if cfg.dust is not None else sun
    # Nap: az AP-k csak teljesen a korong belsejében (a perem mentén csak a peremre merőleges
    # eltolás mérhető megbízhatóan – apertúra-probléma –, a fényudvar pedig kockánként változik)
    ap_erode, ap_cover = (12, 0.999) if sun else (4, 0.7)
    if sun and cfg.mineral > 0:
        _log("Nap-mód: az ásványi hold (mineral) színkiemelés itt nem értelmezett, kihagyva", t0)
        cfg.mineral = 0.0

    # ---- 1. elemzés: kicsinyített kép, Hold-maszk, globális élesség, telítettség -----------------
    def analyse(i, raw):
        # gyors út: szürke uint8-ból, a teljes felbontású munka csak a Hold körüli kivágáson
        scale = 1.0 / (65535.0 if raw.dtype == np.uint16 else 255.0)
        g8 = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        s = cv2.resize(g8, (W // ds, H // ds), interpolation=cv2.INTER_AREA).astype(np.float32) * scale
        m = geo.moon_mask(s, 0.4 if sun else 0.25)
        if sun and m.any():
            m = geo.circle_mask(m)
        if not m.any():
            return i, s.astype(np.float16), m, 0.0, 0.0, None
        ys, xs = np.nonzero(m)
        hs_, ws_ = m.shape
        sy0, sy1 = max(0, ys.min() - 3), min(hs_, ys.max() + 4)
        sx0, sx1 = max(0, xs.min() - 3), min(ws_, xs.max() + 4)
        me = geo.erode(m, 6)[sy0:sy1, sx0:sx1]
        mfull = cv2.resize(me.astype(np.uint8), ((sx1 - sx0) * ds, (sy1 - sy0) * ds),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        y0, x0 = sy0 * ds, sx0 * ds
        g = g8[y0:y0 + mfull.shape[0], x0:x0 + mfull.shape[1]].astype(np.float32) * scale
        mfull = mfull[:g.shape[0], :g.shape[1]]
        rc = raw[y0:y0 + g.shape[0], x0:x0 + g.shape[1]]
        mx_ = rc.max(axis=2) if rc.ndim == 3 else rc
        sat = float((mx_[mfull] >= 0.98 / scale).mean()) if mfull.any() else 0.0
        q = geo.sharpness(g, mfull)
        if sun:
            # Nap: a perem élessége is (a felszíni mérték kevés részletnél főleg zajt mér)
            band_s = (geo.dilate(m, 2) & ~geo.erode(m, 2))[sy0:sy1, sx0:sx1]
            band = cv2.resize(band_s.astype(np.uint8), (mfull.shape[1], mfull.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            q = (q, geo.limb_sharpness(g, band))
            # napfoltok helye (a globális illesztés ezekre igazít), teljes képkocka-koordinátában
            spots = [(x + x0, y + y0, a, d) for x, y, a, d in geo.find_spots(g, mfull)]
            return i, s.astype(np.float16), m, q, sat, spots
        return i, s.astype(np.float16), m, q, sat, None

    smalls, masks, sharp, satin, spots = {}, {}, {}, {}, {}
    for i, s, m, q, st, sp in _pmap(src, None, analyse, workers, label="elemzés"):
        smalls[i], masks[i], sharp[i], satin[i], spots[i] = s, m, q, st, sp
    n = len(sharp)
    if sun:
        # Nap: felszíni részlet × perem-élesség, mindkettő a mediánjára normálva (mértani közép)
        det = np.array([sharp[i][0] if isinstance(sharp[i], tuple) else 0.0 for i in range(n)])
        lim = np.array([sharp[i][1] if isinstance(sharp[i], tuple) else 0.0 for i in range(n)])
        md, ml = np.median(det[det > 0]) if (det > 0).any() else 1, np.median(lim[lim > 0]) if (lim > 0).any() else 1
        for i in range(n):
            sharp[i] = float(np.sqrt(max(det[i] / md, 0) * max(lim[i] / ml, 0)))
    ts = src.timestamps or list(range(n))
    names = src.names()[:n]
    _log(f"elemzés kész: {n} kocka ({'Nap' if sun else 'Hold'}); beégett pixelek a bemeneten: "
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
        h, c, m = geo.align_one(ref_small, masks[ref_i], ref_kp, ref_desc, smalls[i].astype(np.float32), masks[i],
                                sun=sun, spots_ref=spots.get(ref_i), spots_frm=spots.get(i), ds=ds)
        return i, h, c, m

    with ThreadPoolExecutor(workers) as ex:
        stepa = max(1, len(cand) // 10)
        for k, (i, h, c, m) in enumerate(ex.map(align, cand), 1):
            Hs_small[i], ecc[i], meth[i] = h, c, m
            if k % stepa == 0 or k == len(cand):
                print(f"[{time.time() - t0:7.1f}s]   globális illesztés: {k}/{len(cand)}  PROGRESS {k}/{len(cand)}",
                      flush=True)
    nweak = sum(1 for i in cand if not (meth[i].endswith(("+ecc", "+ecc_t")) or meth[i].startswith("sun_spots"))
                and i != ref_i)
    if sun:
        from collections import Counter
        cnt = Counter("sun_spots_sim" if m_.startswith("sun_spots_sim") else m_ for m_ in meth.values())
        _log("Nap-illesztés módja: " + ", ".join(f"{k}: {v}" for k, v in cnt.most_common())
             + f" (napfolt a referencián: {len(spots.get(ref_i) or [])})", t0)
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

    def corrected(i, raw, ca=False):
        """Flat + erősítés. A CA-korrekciót a stack a remap-be olvasztva végzi (ca=False);
        ca=True csak külön kérésre, teljes képre."""
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

    # Kimeneti vászon: kezdetben a teljes képkocka, a kezdeti referencia után a Hold körüli kivágás
    # (ROI). A vászon (0,0) pontja a referencia-képkocka off=(x0,y0) pontja; ezt a T eltolás a
    # leképezésekbe olvasztja, így minden további lépés csak a ROI-n dolgozik.
    cv = dict(W=W, H=H, off=(0, 0), cg=distmod.CoarseGrid(W, H, 8))

    def T_off():
        x0, y0 = cv["off"]
        return np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)

    def warp_gray(i, raw):
        g = luminance(corrected(i, raw))
        return i, g, maps[i].warp(g, cv["W"], cv["H"], cv["cg"], cv2.INTER_LINEAR, -1.0)

    nref = min(len(cand), max(cfg.ref_min, int(math.ceil(cfg.ref_percent / 100 * len(cand)))))
    ref_set = [i for i in order if i in Hf][:nref]
    ref_mask_d = cv2.resize(masks[ref_i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    def build_reference():
        OW, OH = cv["W"], cv["H"]
        acc = np.zeros((OH, OW), np.float64); cnt = np.zeros((OH, OW), np.float64)
        for i, g, w in _pmap(src, ref_set, warp_gray, workers, label="referencia"):
            v = w >= 0
            acc[v] += w[v]; cnt[v] += 1
        ref = (acc / np.maximum(cnt, 1)).astype(np.float32)
        rm = maps[ref_i].warp(ref_mask_d, OW, OH, cv["cg"], cv2.INTER_NEAREST, 0.0) > 0.5
        rm &= cnt >= max(1, nref // 2)
        aps, struct, step = apmod.make_grid(ref, geo.erode(rm, ap_erode), cfg.half_box, cfg.step, min_cover=ap_cover,
                                            struct_rel=cfg.struct_rel)
        return ref, rm, aps, step

    def full_mask(m):
        """ROI-maszk visszahelyezése a teljes képkocka-méretbe."""
        out = np.zeros((H, W), bool)
        x0, y0 = cv["off"]
        out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        return out

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

        if cfg.processes and len(frames) >= 20 and nproc > 1:
            ctx = dict(inv_g=inv_g, gains=gains, maps={i: maps[i] for i in frames}, W=cv["W"], H=cv["H"],
                       cg=cv["cg"], refb=refb, aps=aps, hb=cfg.half_box, search=cfg.search, min_corr=cfg.min_corr)
            it = _ppmap(src, frames, ctx, nproc, label)
        else:
            it = _pmap(src, frames, one, workers, label=label)
        for i, s, o, qq in it:
            k = pos[i]
            sh[k], ok[k], q[k] = s, o, qq
        return sh, ok, q

    # ---- 4. kezdeti referencia és AP-rács (torzított térben) -----------------------------------
    ref0, ref_mask, aps, step = build_reference()
    # ROI: a Hold befoglaló téglalapja + tartalék (keresés, AP-box, torzítás-korrekció)
    ys_, xs_ = np.nonzero(ref_mask)
    margin = 2 * (cfg.search + cfg.half_box) + 120
    x0, x1 = max(0, xs_.min() - margin), min(W, xs_.max() + 1 + margin)
    y0, y1 = max(0, ys_.min() - margin), min(H, ys_.max() + 1 + margin)
    cv.update(W=int(x1 - x0), H=int(y1 - y0), off=(int(x0), int(y0)))
    cv["cg"] = distmod.CoarseGrid(cv["W"], cv["H"], 8)
    maps = {i: distmod.FrameMap(Hf[i] @ T_off()) for i in cand}
    ref0, ref_mask, aps, step = build_reference()
    _log(f"kezdeti referencia: {nref} kocka átlaga; munkaterület (ROI) {cv['W']}x{cv['H']} "
         f"({cv['W'] * cv['H'] / (W * H) * 100:.0f}% a teljes képből); AP-rács: {len(aps)} pont", t0)

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
        p = aps[:, ::-1].astype(np.float64)  # (x, y) a ROI-ban
        off = np.array(cv["off"], np.float64)
        for k, i in enumerate(dframes):
            o = ok0[k]
            xr = p[o] + off                                   # teljes ref-képkocka koordináta
            xk = maps[i].points(p[o] + sh0[k][o][:, ::-1])    # képkocka-koordináta
            corr.append((xr, xk))
        dist, dinfo = distmod.calibrate(corr, W, H, deg=cfg.dist_deg)
        if dist is not None:
            dist.dense_D()  # a sűrű D-táblázat egyszer, előre (a szálak ne számolják párhuzamosan)
            ys, xs = np.nonzero(geo.erode(ref_mask, 20))
            sel = np.random.default_rng(1).choice(len(xs), min(4000, len(xs)), replace=False)
            pts_ref = np.stack([xs[sel], ys[sel]], 1).astype(np.float64) + off
            maps = {i: distmod.FrameMap(distmod.frame_homography(dist, Hf[i], pts_ref) @ T_off(), dist)
                    for i in cand}
            fm = full_mask(ref_mask)
            dinfo["max_shift_moon_px"] = distmod.max_shift_in(dist, fm)
            cv2.imwrite(os.path.join(outdir, "_torzitas_mezo.png"), distmod.draw_field(dist, W, H, fm))
            _log(f"torzítás: {dinfo['frames']} kocka, {dinfo['points']} pontpár; maradék rms "
                 f"{dinfo['rms_before']:.2f} -> {dinfo['rms_after']:.2f} px (medián {dinfo['median_before']:.2f} -> "
                 f"{dinfo['median_after']:.2f}); max korrekció a Hold területén {dinfo['max_shift_moon_px']:.1f} px", t0)
            ref0, ref_mask, aps, step = build_reference()
            _log(f"referencia torzítatlan térben újraépítve; AP-rács: {len(aps)} pont", t0)
        else:
            _log(f"torzítás: nem alkalmazva ({dinfo.get('msg')}"
                 + (f", rms {dinfo['rms_before']:.2f} -> {dinfo['rms_after']:.2f} px" if 'rms_before' in dinfo else "")
                 + ")", t0)

    # ---- 5a. szenzorhoz rögzített porszemek (Napnál alapból) ------------------------------------
    dust_info = dict(enabled=False, msg="kikapcsolva")
    if use_dust:
        from . import dust as dustmod
        k = min(30, len(cand))
        dsel = [cand[j] for j in np.unique(np.round(np.linspace(0, len(cand) - 1, k)).astype(int))]
        dmap, dust_info = dustmod.estimate(src, dsel, corrected, maps, ref0, ref_mask, W, H, masks, ds,
                                           log=lambda s_: _log(s_, t0))
        if dmap is not None:
            inv_g = (inv_g if inv_g is not None else np.ones((H, W), np.float32)) / dmap
            cv2.imwrite(os.path.join(outdir, "_pormaszk.png"),
                        np.clip(dmap * 255, 0, 255).astype(np.uint8))
            _log(f"pormaszk: {dust_info['specks']} porszem/hiba a szenzoron (legmélyebb "
                 f"{dust_info['max_depth'] * 100:.0f}%), minden kockán korrigálva", t0)
            ref0, ref_mask, aps, step = build_reference()
            _log(f"referencia pormentesen újraépítve; AP-rács: {len(aps)} pont", t0)
        else:
            _log(f"pormaszk: nincs korrekció ({dust_info.get('msg')})", t0)

    # ---- 5b. próbafuttatásos kalibráció (AP-box × kiválasztási arány) ---------------------------
    trial_info = None
    if cfg.trial or cfg.trial_only:
        from . import trial as trialmod
        if len(cand) > cfg.trial_frames:
            tsel = [cand[j] for j in np.unique(np.round(np.linspace(0, len(cand) - 1, cfg.trial_frames)).astype(int))]
        else:
            tsel = list(cand)
        sp_xy = None
        if sun and spots.get(ref_i):
            # a referencia napfoltjai a kimeneti (ROI) koordinátákban
            px_, py_ = maps[ref_i].inv_map_xy(np.array([s_[0] for s_ in spots[ref_i]]),
                                              np.array([s_[1] for s_ in spots[ref_i]]))
            sp_xy = [(float(x_), float(y_), s_[2]) for x_, y_, s_ in zip(px_, py_, spots[ref_i])]
        # Napnál kisebb csempe a foltok körül (a nagy, egyenletes felszín ne nyomja el a foltok részletét)
        ty0, tx0, th, tw = trialmod.pick_tile(ref0, ref_mask, min(cfg.trial_tile, 640) if sp_xy else cfg.trial_tile,
                                              sp_xy)
        tiles = {}
        for i, g, w in _pmap(src, tsel, warp_gray, workers, label="kalibrációs kockák"):
            tiles[i] = w[ty0:ty0 + th, tx0:tx0 + tw].copy()
        tl = [tiles[i] for i in tsel]
        tgs = np.array([sharp[i] for i in tsel])
        ys_m, xs_m = np.nonzero(ref_mask)
        moon_d = max(ys_m.max() - ys_m.min(), xs_m.max() - xs_m.min())
        hbs = {24, 32, 48, int(cfg.half_box)}
        if moon_d > 2400:          # nagy (pl. digitálisan nagyított) Hold: nagyobb box is jelölt
            hbs.add(64)
        hbs = sorted(h for h in hbs if 8 * h <= min(th, tw))
        pcts = sorted({10.0, 20.0, 35.0, float(cfg.stack_percent)}) if len(cand) > 60 else \
            sorted({30.0, 50.0, 80.0, float(cfg.stack_percent)})
        tmin = cfg.min_frames if len(cand) <= cfg.trial_frames else \
            max(3, int(round(cfg.min_frames * len(tsel) / len(cand))))
        _log(f"próbafuttatásos kalibráció: csempe {tw}x{th} px, {len(tl)} kocka, "
             f"AP-box {[2 * h for h in hbs]} px × legjobb {pcts} %", t0)
        targs = (ref0[ty0:ty0 + th, tx0:tx0 + tw], ref_mask[ty0:ty0 + th, tx0:tx0 + tw], tgs)
        best, trials = trialmod.run_trials(tl, *targs, hbs, pcts, cfg.search, tmin, cfg.struct_rel, cfg.min_corr,
                                           log=lambda s: _log(s, t0), ap_erode=ap_erode, ap_cover=ap_cover)
        # ha a győztes a jelöltlista szélén van, abba az irányba tovább próbálunk (legfeljebb 2 kör)
        min_pct = 100.0 * tmin / max(len(tl), 1)
        for _round in range(2):
            if best is None:
                break
            tried = {(r["half_box"], r["percent"]) for r in trials}
            new_hb = []
            hb_all = sorted({r["half_box"] for r in trials})
            if best["half_box"] == hb_all[0] and hb_all[0] > 12:
                new_hb.append(max(12, _r4(best["half_box"] * 0.7)))
            elif best["half_box"] == hb_all[-1] and 8 * _r4(best["half_box"] * 1.4) <= min(th, tw):
                new_hb.append(_r4(best["half_box"] * 1.4))
            new_pct = []
            p_all = sorted({r["percent"] for r in trials})
            if best["percent"] == p_all[0] and p_all[0] > min_pct + 1:
                new_pct.append(max(round(min_pct + 1), round(best["percent"] / 2)))
            elif best["percent"] == p_all[-1] and p_all[-1] < 95:
                new_pct.append(min(100.0, round(best["percent"] * 1.3)))
            combos = [(h, best["percent"]) for h in new_hb] + [(best["half_box"], p) for p in new_pct]
            combos = [c for c in combos if c not in tried]
            if not combos:
                break
            _log(f"a legjobb a jelöltlista szélén van → további próbák: "
                 f"{', '.join(f'box {2 * h} px / {p:g}%' for h, p in combos)}", t0)
            for h, p in combos:
                b2, r2 = trialmod.run_trials(tl, *targs, [h], [p], cfg.search, tmin, cfg.struct_rel,
                                             cfg.min_corr, log=lambda s: _log(s, t0),
                                             ap_erode=ap_erode, ap_cover=ap_cover)
                trials += r2
            best = max(trials, key=lambda r: r["score"])
        if best is not None:
            for r in trials:
                r["relative"] = r["score"] / best["score"]
        del tiles, tl
        if best is not None:
            base_r = next((r for r in trials if r["half_box"] == int(cfg.half_box)
                           and r["percent"] == float(cfg.stack_percent)), None)
            gain = (best["score"] / base_r["score"] - 1) * 100 if base_r else float("nan")
            _log(f"kalibráció eredménye: AP-box {2 * best['half_box']} px, legjobb {best['percent']:g}% "
                 f"(pontszám +{gain:.1f}% a kiinduló beállításhoz képest)", t0)
            trial_info = dict(best=best, trials=trials, tile=[tx0, ty0, tw, th], frames=len(tsel),
                              start=dict(half_box=int(cfg.half_box), percent=float(cfg.stack_percent)),
                              gain_percent=gain)
            if cfg.trial_only:
                base = os.path.splitext(os.path.basename(os.path.normpath(inp)))[0]
                with open(os.path.join(outdir, f"{base}_kalibracio.json"), "w", encoding="utf-8") as fh:
                    json.dump(trial_info, fh, ensure_ascii=False, indent=1)
                _log(f"kész (csak kalibráció): {os.path.join(outdir, base + '_kalibracio.json')}", t0)
                return dict(trial=trial_info, seconds=time.time() - t0)
            if best["half_box"] != cfg.half_box:
                cfg.half_box = best["half_box"]
                aps, _s, step = apmod.make_grid(ref0, geo.erode(ref_mask, ap_erode), cfg.half_box, cfg.step, min_cover=ap_cover,
                                                struct_rel=cfg.struct_rel)
            cfg.stack_percent = best["percent"]
        else:
            _log("kalibráció: nem volt értékelhető próba, marad a kiinduló beállítás", t0)

    OW, OH = cv["W"], cv["H"]
    nb = apmod.Neighbors(aps, step)
    field = apmod.Field(aps, (OH, OW), step)
    scg = distmod.CoarseGrid(OW, OH, field.f)   # ugyanaz a durva rács, mint az eltolási mezőé
    cand_pos = {i: k for k, i in enumerate(cand)}
    _log(f"AP-rács: {len(aps)} pont, box={2 * cfg.half_box}px, lépés={step}px", t0)

    # ---- 5c. Nap: peremillesztés --------------------------------------------------------------------
    # A globális illesztés a napfoltokra igazít; a látómező torzítása miatt a korong pereme ehhez
    # képest kockánként eltolódhat/átméreteződhet (valódi 146 s-os videón ~20 px) → elkent, kettős perem.
    # Kockánként ismert a perem (körillesztés), így egy sima, sugárirányú korrekció a kocka peremét a
    # referencia peremére húzza; a súly a középpont felé ρ^6 szerint elhalványul (a foltoknál elhanyagolható).
    limb = None
    if sun:
        rp = geo.disc_params(ref_mask)
        if rp is not None:
            crx, cry, Rr = rp
            limb = {}
            for i in cand:
                dp = geo.disc_params(masks[i])
                if dp is None:
                    continue
                cfx, cfy = dp[0] * ds + (ds - 1) / 2, dp[1] * ds + (ds - 1) / 2
                ox, oy = maps[i].inv_map_xy(np.array([cfx]), np.array([cfy]))
                sk = float(np.sqrt(abs(np.linalg.det(maps[i].G[:2, :2]))))
                limb[i] = (float(ox[0]), float(oy[0]), dp[2] * ds / max(sk, 1e-6))
            dev = [np.hypot(v[0] - crx, v[1] - cry) + abs(v[2] - Rr) for v in limb.values()]
            _log(f"Nap: peremillesztés – a perem eltérése a foltokhoz igazított kockákon: medián "
                 f"{np.median(dev):.1f} px, max {np.max(dev):.1f} px", t0)

    limb_ref = geo.disc_params(ref_mask) if limb is not None else None

    def limb_shift(i, x, y):
        crx, cry, Rr = limb_ref
        ckx, cky, Rk = limb[i]
        rho = np.hypot(x - crx, y - cry) / Rr
        w = np.clip(rho, 0, 1.1) ** 6
        dx = (ckx - crx) + (x - crx) * (Rk / Rr - 1)
        dy = (cky - cry) + (y - cry) * (Rk / Rr - 1)
        return dx * w, dy * w

    # ---- 6. stack (warp-mező) -----------------------------------------------------------------------
    def stack(sh, ok, wap, wglob, color, interp=cv2.INTER_CUBIC, label="stack"):
        need = [i for i in cand if wap[cand_pos[i]].max() > 0 or wglob[cand_pos[i]] > 0]
        ch = 3 if color else 1
        num = np.zeros((OH, OW, ch), np.float32); den = np.zeros((OH, OW), np.float32)
        caf = ca_holder[0] if color else None

        def one(i, raw):
            k = cand_pos[i]
            f = corrected(i, raw)
            img = f if color else luminance(f)
            # minden leképezés a durva rácson (eltolási mező + homográfia + torzítás [+ CA]), majd felskálázás
            xs = scg.xc + field.shift(sh[k, :, 1], ok[k])
            ys = scg.yc + field.shift(sh[k, :, 0], ok[k])
            if limb is not None and i in limb:
                dlx, dly = limb_shift(i, scg.xc, scg.yc)
                xs = xs + dlx; ys = ys + dly
            mxc, myc = maps[i].map_xy(xs, ys)
            mx, my = scg.up_map(mxc, myc)
            wm = field.up(field.weight(wap[k], wglob[k]))
            if caf is not None:
                out = np.empty((OH, OW, 3), np.float32)
                out[..., colormod.G] = cv2.remap(img[..., colormod.G], mx, my, interp, borderMode=cv2.BORDER_REFLECT)
                for c in (colormod.R, colormod.B):
                    dxc, dyc = caf.shift_at(c, mxc, myc)
                    cx, cy = scg.up_map(mxc + dxc, myc + dyc)
                    out[..., c] = cv2.remap(img[..., c], cx, cy, interp, borderMode=cv2.BORDER_REFLECT)
            else:
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
    # Megbízhatatlan AP-k elhagyása: amely AP a kockák negyedében sem mér megbízhatóan (pl. a beégett,
    # egyenletes napkorongon), az csak zajt adna a helyi mezőhöz és a súlyokhoz. A helyén a globális
    # (egész kockás) élességi rangsor dönt. (Holdon az AP-k ~98%-a megbízható, ott ez nem változtat.)
    rel = ok.mean(0)
    keep = rel >= 0.25
    if keep.sum() < 3:
        keep = np.zeros_like(keep); keep[np.argsort(-rel)[:3]] = True
    if keep.sum() < len(aps):
        n_before = len(aps)
        aps = aps[keep]; sh = sh[:, keep]; ok = ok[:, keep]; q = q[:, keep]; raw_ok = raw_ok[:, keep]
        nb = apmod.Neighbors(aps, step)
        field = apmod.Field(aps, (OH, OW), step)
        scg = distmod.CoarseGrid(OW, OH, field.f)
        _log(f"megbízhatatlan AP-k elhagyva: {n_before - len(aps)} db; marad {len(aps)} "
             f"(megbízható {ok.mean() * 100:.1f}%)", t0)

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
    y0, y1 = max(0, ys.min() - pad), min(OH, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(OW, xs.max() + pad)
    # maradék CA + légköri diszperzió a kész stacken (kimeneti koordinátákban, lineáris mező)
    ca2 = dict(enabled=False, msg="kikapcsolva")
    if cfg.ca and result.ndim == 3:
        s2 = colormod.measure(np.clip(result, 0, None), geo.erode(ref_mask, 30), step=56)
        f2, ca2 = colormod.CAField.fit(s2, OW, OH, deg=1, min_pts=30)
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
    if sun and cfg.sun_enhance:
        enh = sun_enhance(lin, cm)
        save16(os.path.join(outdir, f"{base}_nap_kiemelt.tif"), enh)
        save8(os.path.join(outdir, f"{base}_nap_kiemelt.png"), enh)
        _log("Nap: szélsötétedés-kiegyenlített, kiemelt változat elkészült (_nap_kiemelt)", t0)
    if os.path.exists(os.path.join(outdir, "_pormaszk.png")):
        os.replace(os.path.join(outdir, "_pormaszk.png"), os.path.join(outdir, f"{base}_pormaszk.png"))
    _log(f"fényesség: beégett pixel a célponton — bemenet max {bright['input_saturated_max'] * 100:.3f}%, "
         f"stack {bright['stack_saturated'] * 100:.3f}%, élesített {bright['sharpened_saturated'] * 100:.3f}%", t0)

    vis = cv2.cvtColor(np.clip(ref0 * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rel = ok.mean(0)
    for (y, x), r in zip(aps, rel):
        col = (0, int(255 * r), int(255 * (1 - r)))
        cv2.rectangle(vis, (int(x - cfg.half_box), int(y - cfg.half_box)),
                      (int(x + cfg.half_box), int(y + cfg.half_box)), col, 2)
    cv2.imwrite(os.path.join(outdir, f"{base}_aps.jpg"), cv2.resize(vis, (OW // 2, OH // 2)),
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
        trial_calibration=trial_info, target=cfg.target,
        dust={k: v for k, v in dust_info.items() if k != "speck_list"} | (
            {"specks_top": dust_info.get("speck_list", [])[:10]} if dust_info.get("enabled") else {}),
        chromatic_aberration=dict(per_frame=cainfo, residual_on_stack=ca2),
        frames_detail=[dict(name=names[i], t=float(ts[i]) if i < len(ts) else None, sharp=sharp[i],
                            saturated=satin[i], candidate=i in cand_pos, ecc=ecc.get(i), method=meth.get(i),
                            gain=gains.get(i), global_weight=float(wglob[cand_pos[i]]) if i in cand_pos else 0.0,
                            ap_weight_mean=float(wsel[cand_pos[i]].mean()) if i in cand_pos else 0.0)
                       for i in range(n)],
        roi=[cv["off"][0], cv["off"][1], OW, OH],
        crop=[int(x0) + cv["off"][0], int(y0) + cv["off"][1], int(x1) + cv["off"][0], int(y1) + cv["off"][1]],
        seconds=time.time() - t0)
    with open(os.path.join(outdir, f"{base}_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=float)
    _log(f"kész: {outdir}", t0)
    return report
