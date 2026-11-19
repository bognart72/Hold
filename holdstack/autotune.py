"""Paraméter-hangolás az adott felvételhez.

Gyors előzetes elemzés (videónál ~80 mintakocka, fotóknál mind) és ebből javasolt beállítások,
mindegyikhez indoklással. A feldolgozás alapértékeit NEM írja felül véglegesen: a javaslatot a
GUI a mezőkbe tölti, a parancssor (--auto) pedig az adott futásra alkalmazza.

Mért jellemzők -> paraméterek:
  Hold-átmérő (px)                    -> AP-box mérete
  helyi eltolások (seeing, rezgés)    -> keresési sugár
  élesség-eloszlás, képkockaszám      -> kiválasztási arány, min. kockaszám, jelöltek aránya
  a Hold vándorlása a szenzoron       -> flat- és torzítás-kalibráció be/ki
  színcsatorna-eltolás (CA)           -> CA-korrekció be/ki, színszegély-csillapítás
  beégett pixelek a bemeneten         -> csúcsfény-levágás (knee)
  zajszint és a stackelt kockák száma -> élesítés erőssége
Használat: python -m holdstack.autotune <videó vagy képmappa> [--json kimenet.json]
"""
import sys
import json
import math
import time
import argparse

import numpy as np
import cv2

from .source import Source, to_float, luminance
from . import geometry as geo
from . import ap as apmod
from . import color as colormod

DEFAULTS = dict(percent=20, min_frames=5, candidates=50, half_box=32, search=16, knee=0.85, defringe=0.5,
                sharpen="2.0,1.5,1.2,1.0", refine=True, flat=True, distortion=True, ca=True, gain=True)


def _log(t0, msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _r4(x):
    return int(4 * round(x / 4.0))


def analyse(path, max_samples=80, log=True):
    t0 = time.time()
    src = Source(path)
    H, W = src.shape
    n_total = src.n
    if n_total <= max_samples:
        idx = list(range(n_total))
    else:
        idx = sorted(set(np.round(np.linspace(0, n_total - 1, max_samples)).astype(int).tolist()))
    ds = 4
    rec = {}
    for k, (i, raw) in enumerate(src.frames(idx), 1):
        scale = 1.0 / (65535.0 if raw.dtype == np.uint16 else 255.0)
        g8 = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        s = cv2.resize(g8, (W // ds, H // ds), interpolation=cv2.INTER_AREA).astype(np.float32) * scale
        m = geo.moon_mask(s)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min() * ds, (ys.max() + 1) * ds, xs.min() * ds, (xs.max() + 1) * ds
        me = cv2.resize(geo.erode(m, 6).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        g = g8[y0:y1, x0:x1].astype(np.float32) * scale
        mm = me[y0:y1, x0:x1]
        sharp = geo.sharpness(g, mm)
        # zaj: a nagyfrekvenciás maradék robusztus szórása a Holdon, a fényességhez viszonyítva
        hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
        noise = float(1.4826 * np.median(np.abs(hp[mm]))) / max(float(g[mm].mean()), 1e-3) if mm.any() else 0.0
        rc = raw[y0:y1, x0:x1]
        mxc = rc.max(axis=2) if rc.ndim == 3 else rc
        sat = float((mxc[mm] >= 0.98 / scale).mean()) if mm.any() else 0.0
        diam = max(y1 - y0, x1 - x0)
        cy, cx = ys.mean() * ds, xs.mean() * ds
        rec[i] = dict(small=s, mask=m, sharp=sharp, noise=noise, sat=sat, diam=diam, c=(cx, cy))
        if log and (k % max(1, len(idx) // 5) == 0 or k == len(idx)):
            _log(t0, f"hangolás – mintakockák elemzése: {k}/{len(idx)}  PROGRESS {k}/{len(idx)}")
    if len(rec) < 2:
        raise RuntimeError("A felvételen nem található Hold (vagy túl kevés a használható kocka).")
    keys = sorted(rec)
    sharp = np.array([rec[i]["sharp"] for i in keys])
    best = keys[int(np.argmax(sharp))]

    # ---- helyi eltolások egy kis részhalmazon (nagy keresési sugárral) --------------------------
    sub = [keys[j] for j in np.unique(np.round(np.linspace(0, len(keys) - 1, min(12, len(keys)))).astype(int))]
    sub = [i for i in sub if i != best][:11]
    if log:
        _log(t0, f"hangolás – illesztés és helyi eltolások mérése {len(sub)} kockán…")
    ref_small = rec[best]["small"]
    kp, desc = geo.reference_features(ref_small, rec[best]["mask"])
    Hs = {}
    for i in sub:
        h, c, m = geo.align_one(ref_small, rec[best]["mask"], kp, desc, rec[i]["small"], rec[i]["mask"])
        Hs[i] = geo.scale_h(h, ds)
    diam = float(np.median([rec[i]["diam"] for i in keys]))
    hb_probe = max(16, min(64, _r4(diam / 50)))
    frames = dict(src.frames([best] + sub))
    refg = luminance(to_float(frames[best]))
    refm = cv2.resize(rec[best]["mask"].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    aps, _, step = apmod.make_grid(refg, geo.erode(refm, 8), hb_probe, int(hb_probe * 3), struct_rel=0.1)
    refb = cv2.GaussianBlur(refg, (0, 0), 1.0)
    shifts = []
    probe_search = 40
    for k, i in enumerate(sub, 1):
        if log:
            _log(t0, f"hangolás – helyi eltolások: {k}/{len(sub)}  PROGRESS {k}/{len(sub)}")
        g = luminance(to_float(frames[i]))
        w = cv2.warpPerspective(g, Hs[i], (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        valid = w >= 0
        wb = cv2.GaussianBlur(np.where(valid, w, 0).astype(np.float32), (0, 0), 1.0)
        s, o, _ = apmod.measure(wb, valid, refb, aps, hb_probe, probe_search, 0.5)
        if o.any():
            shifts.append(np.hypot(s[o, 0], s[o, 1]))
    sh = np.concatenate(shifts) if shifts else np.array([4.0])
    if log:
        _log(t0, f"hangolás – helyi eltolások mérve {len(sub)} kockán, {len(sh)} pontban")

    # ---- kromatikus aberráció a legjobb kockán ------------------------------------------------
    ca_p95 = 0.0
    f = to_float(frames[best])
    if f.ndim == 3:
        smp = colormod.measure(f, geo.erode(refm, 30))
        mags = [np.hypot(v[:, 2], v[:, 3]) for v in smp.values() if len(v)]
        if mags:
            ca_p95 = float(np.percentile(np.concatenate(mags), 95))

    cpos = np.array([rec[i]["c"] for i in keys])
    spread = float(np.hypot(*(cpos.std(0) / np.array([W, H]))))
    return dict(kind=src.kind, frames=n_total, sampled=len(keys), size=[W, H], moon_diam_px=diam,
                sharp_rel=(sharp / sharp.max()).tolist(), noise=float(np.median([rec[i]["noise"] for i in keys])),
                sat_max=float(max(rec[i]["sat"] for i in keys)), shift_p50=float(np.median(sh)),
                shift_p95=float(np.percentile(sh, 95)), shift_max=float(np.percentile(sh, 99.5)),
                drift_spread=spread, ca_p95=ca_p95, seconds=time.time() - t0)


def suggest(a):
    """A mért jellemzőkből javasolt paraméterek + indoklás."""
    p = dict(DEFAULTS)
    why = []
    n = a["frames"]
    video = a["kind"] == "video"

    hb = max(16, min(64, _r4(a["moon_diam_px"] / 52)))
    p["half_box"] = hb
    why.append(f"AP-box félszélesség {hb} px: a Hold átmérője ~{a['moon_diam_px']:.0f} px "
               f"(kb. 1/52 része → elég részlet és stabil korreláció)")

    srch = int(max(8, min(48, math.ceil(a["shift_max"] * 1.5) + 4)))
    p["search"] = srch
    why.append(f"keresési sugár {srch} px: a globális illesztés utáni helyi eltolás medián "
               f"{a['shift_p50']:.1f} px, p95 {a['shift_p95']:.1f} px, a legnagyobbak ~{a['shift_max']:.1f} px")

    rel = np.array(a["sharp_rel"])
    good = float((rel >= 0.85).mean())
    if video:
        pct = int(max(10, min(40, round(good * 100 / 5) * 5)))
        p["percent"] = pct
        p["min_frames"] = int(max(5, min(20, round(0.03 * n))))
        p["candidates"] = int(max(30, min(80, 2 * pct + 10)))
        why.append(f"legjobb {pct}%: a mintakockák {good * 100:.0f}%-a a legjobb élesség 85%-án belül van "
                   f"(nagyobb arány = kisebb zaj; kisebb = élesebb, ha a seeing ingadozó)")
        why.append(f"jelöltek {p['candidates']}%, legalább {p['min_frames']} kocka ({n} kockás videó)")
    else:
        pct = int(max(30, min(80, round(good * 100 / 5) * 5)))
        p["percent"] = pct
        p["min_frames"] = int(max(3, min(n, round(0.3 * n))))
        p["candidates"] = 100
        why.append(f"legjobb {pct}%, legalább {p['min_frames']} kép: csak {n} fotó van, a zajcsökkentéshez "
                   f"a jó képek nagy része kell ({good * 100:.0f}% a legjobb élesség 85%-án belül)")

    if a["drift_spread"] < 0.015:
        p["flat"] = False
        p["distortion"] = False
        why.append(f"vinyettálás- és torzítás-kalibráció KI: a Hold alig mozdult a képen "
                   f"(szórás {a['drift_spread'] * 100:.1f}% a képmérethez), nem becsülhető megbízhatóan")
    else:
        why.append(f"vinyettálás- és torzítás-kalibráció BE: a Hold a kép {a['drift_spread'] * 100:.1f}%-ának "
                   f"megfelelő szórással vándorolt (elég a becsléshez)")

    if a["ca_p95"] < 0.15:
        p["defringe"] = 0.3
        why.append(f"színszegély-csillapítás 0.3: a mért színcsatorna-eltolás kicsi (p95 {a['ca_p95']:.2f} px)")
    elif a["ca_p95"] < 0.6:
        p["defringe"] = 0.5
        why.append(f"színszegély-csillapítás 0.5: közepes színcsatorna-eltolás (p95 {a['ca_p95']:.2f} px)")
    else:
        p["defringe"] = 0.7
        why.append(f"színszegély-csillapítás 0.7: erős színcsatorna-eltolás (p95 {a['ca_p95']:.2f} px)")

    if a["sat_max"] > 0.001:
        p["knee"] = 0.80
        why.append(f"csúcsfény-levágás 0.80: a bemeneten van beégés (max {a['sat_max'] * 100:.2f}%), "
                   f"az élesítés ne vigye tovább")
    else:
        why.append("csúcsfény-levágás 0.85: a bemeneten nincs beégés")

    nst = max(p["min_frames"], round(p["percent"] / 100 * n))
    snr = 1.0 / max(a["noise"] / math.sqrt(nst), 1e-4)
    if snr > 400:
        p["sharpen"] = "2.6,1.8,1.3,1.0"; lvl = "erős"
    elif snr > 150:
        p["sharpen"] = "2.0,1.5,1.2,1.0"; lvl = "közepes"
    else:
        p["sharpen"] = "1.5,1.3,1.1,1.0"; lvl = "visszafogott"
    why.append(f"élesítés {lvl} ({p['sharpen']}): becsült zaj a stackben ~1/{snr:.0f} "
               f"(~{nst} kocka átlagolva)")
    return p, why


def main():
    ap_ = argparse.ArgumentParser(prog="holdstack.autotune", description="Paraméter-hangolás a felvételhez")
    ap_.add_argument("input")
    ap_.add_argument("--json", help="a javaslat mentése ide (JSON)")
    a = ap_.parse_args()
    t0 = time.time()
    _log(t0, f"hangolás: {a.input}")
    m = analyse(a.input)
    p, why = suggest(m)
    _log(t0, "javasolt beállítások:")
    for w in why:
        print("    - " + w, flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(dict(params=p, reasons=why, measured=m), fh, ensure_ascii=False, indent=1)
    _log(t0, "hangolás kész")


if __name__ == "__main__":
    main()
