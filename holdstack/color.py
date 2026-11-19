"""Színkezelés: kromatikus aberráció (CA) mérése és korrekciója, színszegély-csillapítás, „mineral moon”.

Laterális CA: a piros és kék csatorna a zöldhöz képest helyfüggően el van tolva (a lencse/okulár
miatt, a szenzorhoz kötötten; tengelyen kívüli, ferde telefonnál aszimmetrikusan). A légköri
diszperzió ehhez egy (közel) állandó, függőleges eltolást ad.

Mérés: rácspontokon a zöld csatorna kis ablakát keressük a piros/kék csatornában (zero-mean NCC,
szubpixeles parabola). Több képkocka méréseiből robusztus 2D polinom-mezőt illesztünk a
szenzor-koordináták függvényében, és ezzel MINDEN képkockát korrigálunk a stack előtt
(így a vándorló Hold különböző helyzeteinek színhibája nem kenődik össze).
A stack után egy második, kisebb (lineáris) korrekció a maradékot és a légköri diszperziót veszi ki.
"""
import numpy as np
import cv2

B, G, R = 0, 1, 2


def _terms(deg):
    return [(i, d - i) for d in range(0, deg + 1) for i in range(d + 1)]


def _basis(x, y, W, H, deg):
    s = max(W, H) / 2.0
    a = (x - W / 2.0) / s
    b = (y - H / 2.0) / s
    return np.stack([a ** i * b ** j for i, j in _terms(deg)], -1)


def measure(img, mask, step=72, hb=20, search=4, min_corr=0.85, min_contrast=0.02):
    """Visszaad: dict ch -> (N,4) tömb [x, y, dx, dy]: a zöld tartalma a csatornában (x+dx, y+dy)-nál van."""
    H, W = img.shape[:2]
    blur = [cv2.GaussianBlur(img[..., c], (0, 0), 0.8) for c in range(3)]
    gg = blur[G]
    out = {R: [], B: []}
    s = search
    for y in range(hb + s, H - hb - s, step):
        for x in range(hb + s, W - hb - s, step):
            if not mask[y, x]:
                continue
            tpl = gg[y - hb:y + hb, x - hb:x + hb]
            if tpl.std() < min_contrast * max(tpl.mean(), 1e-3):
                continue
            for c in (R, B):
                reg = blur[c][y - hb - s:y + hb + s, x - hb - s:x + hb + s]
                res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, (px, py) = cv2.minMaxLoc(res)
                if mx < min_corr or px <= 0 or py <= 0 or px >= 2 * s or py >= 2 * s:
                    continue
                l, cc, r = res[py, px - 1], res[py, px], res[py, px + 1]
                ddx = 0.5 * (l - r) / (l - 2 * cc + r) if (l - 2 * cc + r) < 0 else 0.0
                u, d = res[py - 1, px], res[py + 1, px]
                ddy = 0.5 * (u - d) / (u - 2 * cc + d) if (u - 2 * cc + d) < 0 else 0.0
                out[c].append((x, y, px - s + ddx, py - s + ddy))
    return {c: np.array(v, np.float64).reshape(-1, 4) for c, v in out.items()}


class CAField:
    """Csatornánkénti eltolási polinom-mező (R és B a G-hez képest)."""

    def __init__(self, W, H, deg, coef):
        self.W, self.H, self.deg = W, H, deg
        self.coef = coef  # {ch: (2, M)}
        self._maps = None

    @staticmethod
    def fit(samples, W, H, deg=2, min_pts=40):
        coef, info = {}, {}
        for c, P in samples.items():
            name = "R" if c == R else "B"
            if len(P) < min_pts:
                return None, dict(enabled=False, msg=f"kevés mérőpont ({name}: {len(P)})")
            A = _basis(P[:, 0], P[:, 1], W, H, deg)
            w = np.ones(len(P))
            cc = np.zeros((2, A.shape[1]))
            for _ in range(8):
                sw = np.sqrt(w)[:, None]
                for k in range(2):
                    cc[k] = np.linalg.lstsq(A * sw, P[:, 2 + k] * sw[:, 0], rcond=None)[0]
                res = np.hypot(P[:, 2] - A @ cc[0], P[:, 3] - A @ cc[1])
                mad = np.median(res) * 1.4826 + 1e-3
                w = np.minimum(1.0, 2.0 * mad / (res + 1e-9))
            coef[c] = cc
            pred = np.stack([A @ cc[0], A @ cc[1]], 1)
            info[name] = dict(points=int(len(P)), mean_shift_px=float(np.hypot(*pred.mean(0))),
                              max_shift_px=float(np.hypot(pred[:, 0], pred[:, 1]).max()),
                              residual_px=float(np.median(res)))
        return CAField(W, H, deg, coef), dict(enabled=True, deg=deg, **info, msg="ok")

    def maps(self):
        if self._maps is None:
            ys, xs = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
            A = None
            self._maps = {}
            for c, cc in self.coef.items():
                dx = np.zeros((self.H, self.W), np.float32); dy = np.zeros_like(dx)
                s = max(self.W, self.H) / 2.0
                a = (xs - self.W / 2.0) / s; b = (ys - self.H / 2.0) / s
                for (i, j), cx, cy in zip(_terms(self.deg), cc[0], cc[1]):
                    m = a ** i * b ** j
                    dx += np.float32(cx) * m; dy += np.float32(cy) * m
                self._maps[c] = (xs + dx, ys + dy)
        return self._maps

    def shift_at(self, c, x, y):
        """A c csatorna eltolása (dx, dy) tetszőleges koordináta-tömbökön."""
        s = max(self.W, self.H) / 2.0
        a = (x - self.W / 2.0) / s; b = (y - self.H / 2.0) / s
        cc = self.coef[c]
        dx = np.zeros_like(a); dy = np.zeros_like(a)
        for (i, j), cx, cy in zip(_terms(self.deg), cc[0], cc[1]):
            m = a ** i * b ** j
            dx += cx * m; dy += cy * m
        return dx, dy

    def apply(self, img):
        out = img.copy()
        for c, (mx, my) in self.maps().items():
            out[..., c] = cv2.remap(img[..., c], mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        return out


def defringe(img, strength=0.5, sigma=1.2):
    """Színszegély-csillapítás: az erős fényesség-éleknél a krominanciát a környezeti átlag felé húzza,
    és a krominanciát enyhén simítja (a fényesség érintetlen)."""
    if strength <= 0:
        return img
    ycc = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YCrCb)
    Y = ycc[..., 0]
    gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0); gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1)
    e = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.5)
    e = np.clip(e / (np.percentile(e, 99.5) + 1e-6), 0, 1)
    w = (strength * e)[..., None]
    chroma = ycc[..., 1:]
    smooth = cv2.GaussianBlur(chroma, (0, 0), sigma)
    wide = cv2.GaussianBlur(chroma, (0, 0), 6 * sigma)
    ycc[..., 1:] = (1 - w) * smooth + w * wide
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), 0, None)


def mineral_moon(img, strength=4.0, mask=None, chroma_sigma=2.5, luminance=None, gradient_deg=2):
    """„Mineral moon”: a fényesség (L) változatlan (vagy a megadott, pl. élesített fényesség), a
    zajmentesített színinformációból levonjuk a nagy léptékű színátmenetet (telefonos szín-
    vinyettálás, fehéregyensúly, légkör: sima 2D polinom), majd strength-szeresére erősítjük.
    A valódi ásványi színkülönbségek (kék titánban gazdag, narancs vasban gazdag mare-k)
    regionálisak, ezeket a polinom nem viszi el."""
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)
    L = lab[..., 0]
    # lágy súly a fényességből: a sötét égen és a terminátor árnyékában 0, a megvilágított felszínen 1
    ref = np.median(L[mask]) if mask is not None and mask.any() else np.percentile(L, 90)
    w0 = np.clip((L - 0.08 * ref) / (0.30 * ref), 0, 1)
    # a perem (limbus) mentén a szín megbízhatatlan (színhiba, légköri diszperzió, háttér-bekeverés)
    lit = (w0 > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(lit, cv2.DIST_L2, 5)
    diam = 2.0 * np.sqrt(lit.sum() / np.pi)          # a megvilágított rész egyenértékű átmérője
    r0, r1 = 0.006 * diam, 0.03 * diam               # e sávban halványul a szín a perem felé
    edge = np.clip((dist - r0) / max(r1 - r0, 1.0), 0, 1)
    w = cv2.GaussianBlur(w0 * edge, (0, 0), 4)
    # normált konvolúció: a színsimításba csak a megvilágított felszín szóljon bele
    ab = cv2.GaussianBlur(lab[..., 1:] * w[..., None], (0, 0), chroma_sigma) / \
        (cv2.GaussianBlur(w, (0, 0), chroma_sigma)[..., None] + 1e-4)
    # nagy léptékű színátmenet levonása (súlyozott polinom-illesztés, ritkított mintán)
    H, W = L.shape
    ys, xs = np.mgrid[0:H:8, 0:W:8]
    ws = w[::8, ::8].ravel()
    sel = ws > 0.5
    A = _basis(xs.ravel()[sel].astype(np.float64), ys.ravel()[sel].astype(np.float64), W, H, gradient_deg)
    Af = None
    yy, xx = np.mgrid[0:H, 0:W]
    for k in range(2):
        v = ab[::8, ::8, k].ravel()[sel]
        c = np.linalg.lstsq(A * ws[sel][:, None], v * ws[sel], rcond=None)[0]
        if Af is None:
            Af = _basis(xx.astype(np.float32), yy.astype(np.float32), W, H, gradient_deg)
        ab[..., k] -= (Af @ c.astype(np.float32)).astype(np.float32)
    ab *= strength
    if luminance is not None:
        lab[..., 0] = cv2.cvtColor(np.clip(luminance, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)[..., 0]
    lab[..., 1:] = np.clip(ab, -110, 110) * w[..., None]
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)
