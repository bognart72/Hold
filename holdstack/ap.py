"""Alignment pointok (AP): rács, lokális eltolásmérés, simítás, sűrű mezővé alakítás, súlyozás."""
import math
import warnings

import numpy as np
import cv2
from scipy.spatial import cKDTree


def make_grid(ref_gray, ref_mask, half_box=32, step=0, min_cover=0.7, struct_rel=0.05, min_bright=0.05):
    """Lépcsőzetes AP-rács a referencián. Visszaad: (N,2) int tömb (y, x) és a struktúra-értékek."""
    H, W = ref_gray.shape
    step = step or int(round(1.5 * half_box))
    b = cv2.GaussianBlur(ref_gray, (0, 0), 1.5)
    gy = np.abs(np.diff(b, axis=0))
    gx = np.abs(np.diff(b, axis=1))
    cand, struct = [], []
    hb = half_box
    for r, y in enumerate(range(hb, H - hb, step)):
        x0 = hb + (step // 2 if r % 2 else 0)
        for x in range(x0, W - hb, step):
            mb = ref_mask[y - hb:y + hb, x - hb:x + hb]
            if mb.mean() < min_cover:
                continue
            box = b[y - hb:y + hb, x - hb:x + hb]
            mu = box.mean()
            if mu < min_bright:
                continue
            sy = gy[y - hb:y + hb - 1, x - hb:x + hb].mean() / mu
            sx = gx[y - hb:y + hb, x - hb:x + hb - 1].mean() / mu
            cand.append((y, x)); struct.append(min(sx, sy))
    cand = np.array(cand, int); struct = np.array(struct)
    if len(cand) == 0:
        raise RuntimeError("Nincs alkalmas alignment point (túl kicsi Hold vagy túl nagy AP-méret)")
    keep = struct >= struct_rel * struct.max()
    return cand[keep], struct[keep], step


def measure(warped, valid, ref_blur, aps, half_box, search, min_corr=0.5):
    """Lokális eltolás minden AP-n: a referencia-box keresése a (globálisan illesztett) képkockában.

    warped: a képkocka a referencia-koordinátákban (szürke, ugyanúgy elmosva, mint ref_blur).
    Visszaad: shifts (N,2) [dy,dx] — a ref p pontjának tartalma a warped p+shift helyén van; ok (N,) bool; corr (N,)."""
    N = len(aps)
    shifts = np.zeros((N, 2), np.float32)
    ok = np.zeros(N, bool)
    corr = np.zeros(N, np.float32)
    hb, s = half_box, search
    H, W = warped.shape
    for a, (y, x) in enumerate(aps):
        y0, y1, x0, x1 = y - hb - s, y + hb + s, x - hb - s, x + hb + s
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue
        if not valid[y0:y1, x0:x1].all():
            continue
        tpl = ref_blur[y - hb:y + hb, x - hb:x + hb]
        res = cv2.matchTemplate(warped[y0:y1, x0:x1], tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (px, py) = cv2.minMaxLoc(res)
        corr[a] = mx
        if mx < min_corr or px <= 0 or py <= 0 or px >= 2 * s or py >= 2 * s:
            continue
        # szubpixeles parabola-illesztés
        l, c, r = res[py, px - 1], res[py, px], res[py, px + 1]
        dxs = 0.5 * (l - r) / (l - 2 * c + r) if (l - 2 * c + r) < 0 else 0.0
        u, d = res[py - 1, px], res[py + 1, px]
        dys = 0.5 * (u - d) / (u - 2 * c + d) if (u - 2 * c + d) < 0 else 0.0
        shifts[a] = (py - s + dys, px - s + dxs)
        ok[a] = True
    return shifts, ok, corr


def local_quality_at(gray_frame, pts, half_box):
    """AP-nkénti élesség az EREDETI (nem interpolált) képkockán: Laplace-var / átlag².
    pts: az AP-k helye a képkockában (N,2) [x, y]."""
    gb = cv2.GaussianBlur(gray_frame, (0, 0), 1.5)
    lap = cv2.Laplacian(gb, cv2.CV_32F)
    Hh, Ww = gray_frame.shape
    q = np.zeros(len(pts), np.float32)
    hb = half_box
    for a, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        if yi - hb < 0 or xi - hb < 0 or yi + hb > Hh or xi + hb > Ww:
            continue
        box = gb[yi - hb:yi + hb, xi - hb:xi + hb]
        mu = box.mean()
        if mu > 1e-3:
            q[a] = lap[yi - hb:yi + hb, xi - hb:xi + hb].var() / (mu * mu)
    return q


class Neighbors:
    def __init__(self, aps, step, radius=2.6):
        tree = cKDTree(aps)
        lists = tree.query_ball_point(aps, r=radius * step)
        K = max(len(l) for l in lists)
        self.idx = np.full((len(aps), K), -1, int)
        for a, l in enumerate(lists):
            l = [j for j in l if j != a]
            self.idx[a, :len(l)] = l


def smooth(shifts, ok, nb: Neighbors, tol=1.5):
    """Kiugró AP-eltolások kiszűrése a szomszédok mediánja alapján (képkockánként).
    A kiugrót a mediánra cseréli és megbízhatatlannak jelöli. Visszaad: (shifts, ok)."""
    shifts = shifts.copy(); ok = ok.copy()
    idx = nb.idx
    has = idx >= 0
    for f in range(shifts.shape[0]):
        s = shifts[f]; o = ok[f]
        vals = np.where((has & o[np.maximum(idx, 0)])[..., None], s[np.maximum(idx, 0)], np.nan)
        cnt = np.sum(~np.isnan(vals[..., 0]), axis=1)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(vals, axis=1)
        enough = cnt >= 3
        dev = np.hypot(*(s - med).T)
        bad = enough & o & (dev > tol)
        s[bad] = med[bad]; o[bad] = False
        # sikertelen mérés helyére is a szomszédok mediánja (a mezőhöz), de megbízhatatlan marad
        fill = enough & ~ok[f]
        s[fill] = med[fill]
        shifts[f] = s; ok[f] = o
    return shifts, ok


def soft_select(q, ok, percent, min_frames):
    """AP-nként a legjobb képkockák lágy (cosinus) kiválasztása. q, ok: (F, N). Visszaad: (F, N) súly."""
    F, N = q.shape
    Wt = np.zeros((F, N), np.float32)
    for a in range(N):
        v = np.nonzero(ok[:, a])[0]
        if len(v) == 0:
            continue
        keep = min(len(v), max(min_frames, int(math.ceil(percent / 100.0 * F))))
        order = v[np.argsort(-q[v, a])]
        ranks = np.arange(len(order))
        w = max(1.0, 0.25 * keep)
        t = np.clip((ranks - (keep - 0.5 - w / 2)) / w, 0, 1)
        Wt[order, a] = 0.5 * (1 + np.cos(np.pi * t))
    return Wt


def soft_select_global(score, percent, min_frames):
    F = len(score)
    keep = min(F, max(min_frames, int(math.ceil(percent / 100.0 * F))))
    order = np.argsort(-score)
    ranks = np.empty(F); ranks[order] = np.arange(F)
    w = max(1.0, 0.25 * keep)
    t = np.clip((ranks - (keep - 0.5 - w / 2)) / w, 0, 1)
    return (0.5 * (1 + np.cos(np.pi * t))).astype(np.float32)


class Field:
    """AP-értékekből sima, sűrű mező normált konvolúcióval (durva rácson), majd felskálázás."""

    def __init__(self, aps, shape, step):
        self.H, self.W = shape
        self.f = max(2, step // 6)
        self.hs = math.ceil(self.H / self.f); self.ws = math.ceil(self.W / self.f)
        self.yi = np.clip(np.round((aps[:, 0] - (self.f - 1) / 2) / self.f).astype(int), 0, self.hs - 1)
        self.xi = np.clip(np.round((aps[:, 1] - (self.f - 1) / 2) / self.f).astype(int), 0, self.ws - 1)
        self.sigma = 0.75 * step / self.f
        den = self._conv(np.ones(len(aps), np.float32))
        self.den = den
        self.D0 = float(np.median(den[self.yi, self.xi]))
        self.cover = np.clip(den / self.D0, 0, 1)

    def _conv(self, vals):
        g = np.zeros((self.hs, self.ws), np.float32)
        np.add.at(g, (self.yi, self.xi), vals)
        return cv2.GaussianBlur(g, (0, 0), self.sigma)

    def up(self, coarse):
        big = cv2.resize(coarse, (self.ws * self.f, self.hs * self.f), interpolation=cv2.INTER_LINEAR)
        return big[:self.H, :self.W]

    def shift(self, vals, valid):
        """Eltolási mező (durva rácson); kevés AP-nál 0 felé halványul."""
        w = valid.astype(np.float32)
        num = self._conv(vals * w)
        den = self._conv(w)
        return num / np.maximum(den, 0.5 * self.D0)

    def weight(self, ap_w, glob_w):
        """Súlytérkép (durva rácson): AP-lefedett helyen az AP-súlyok, máshol a globális súly."""
        wap = self._conv(ap_w.astype(np.float32)) / np.maximum(self.den, 1e-6)
        return self.cover * wap + (1 - self.cover) * glob_w
