"""Szenzorhoz rögzített geometriai torzítás önkalibrációja.

Modell:
  u = U(x) = x + s * sum_m c_m * phi_m((x - c) / s)      (torzított szenzor-px -> torzítatlan)
  x = D(u) = U^-1(u)                                     (fixpontos iterációval)
A Hold merev test, így két képkocka torzítatlan koordinátái között egy homográfia áll fenn:
  U(x_k) = G_k( U(x_ref) )
A phi_m a 2..deg fokú monomok (a 0. és 1. fokú, valamint a projektív rész a G_k-ba olvad).
A c együtthatókat "variable projection" módszerrel becsüljük: adott c mellett minden képkockára
zárt alakban illesztünk homográfiát, és a maradékot minimalizáljuk (robusztus, Huber).

A megfelelő pontpárok a már megmért AP-eltolásokból jönnek: a ref p pontja a k. képkockában
H_k(p + d_k(p)) helyen látszik. A Hold driftje miatt a pontok a szenzor különböző részein
látszanak, ebből válik becsülhetővé a torzítás.
"""
import numpy as np
import cv2
from scipy.optimize import least_squares


def _terms(deg):
    return [(i, d - i) for d in range(2, deg + 1) for i in range(d + 1)]


class Distortion:
    def __init__(self, coef, deg, W, H):
        self.coef = np.asarray(coef, np.float64).reshape(2, -1)  # [x-komponens; y-komponens]
        self.deg = deg
        self.W, self.H = W, H
        self.cx, self.cy = W / 2.0, H / 2.0
        self.s = max(W, H) / 2.0
        self.terms = _terms(deg)
        self._dmap = None

    def __getstate__(self):
        # a nagy, újraszámolható D-táblázat ne utazzon a folyamatok között
        st = self.__dict__.copy()
        st["_dmap"] = None
        return st

    def _delta(self, x, y):
        a = (x - self.cx) / self.s
        b = (y - self.cy) / self.s
        dx = np.zeros_like(a); dy = np.zeros_like(a)
        for (i, j), cx_, cy_ in zip(self.terms, self.coef[0], self.coef[1]):
            m = a ** i * b ** j
            dx += cx_ * m; dy += cy_ * m
        return dx * self.s, dy * self.s

    def U(self, pts):
        dx, dy = self._delta(pts[:, 0], pts[:, 1])
        return np.stack([pts[:, 0] + dx, pts[:, 1] + dy], 1)

    def D(self, pts, iters=12):
        x = pts.astype(np.float64).copy()
        for _ in range(iters):
            dx, dy = self._delta(x[:, 0], x[:, 1])
            x = np.stack([pts[:, 0] - dx, pts[:, 1] - dy], 1)
        return x

    def D_xy(self, ux, uy, iters=12):
        """D tetszőleges alakú koordináta-tömbökre."""
        x, y = ux.copy(), uy.copy()
        for _ in range(iters):
            dx, dy = self._delta(x, y)
            x, y = ux - dx, uy - dy
        return x, y

    def dense_D(self):
        """D előre kiszámolva egy kissé kibővített rácson (remap-pel kombinálható)."""
        if self._dmap is None:
            P = int(0.06 * max(self.W, self.H))
            ys, xs = np.mgrid[-P:self.H + P, -P:self.W + P].astype(np.float32)
            x, y = xs.copy(), ys.copy()
            for _ in range(12):
                dx, dy = self._delta(x, y)
                x = (xs - dx).astype(np.float32); y = (ys - dy).astype(np.float32)
            self._dmap = (x, y, P)
        return self._dmap

    def max_shift(self):
        ys, xs = np.mgrid[0:self.H:32, 0:self.W:32].astype(np.float64)
        dx, dy = self._delta(xs, ys)
        return float(np.hypot(dx, dy).max())

    def to_dict(self):
        return dict(deg=self.deg, coef=self.coef.tolist(), max_shift_px=self.max_shift())


class FrameMap:
    """Kimeneti (referencia) koordináta -> képkocka pixel. G: homográfia (torzítatlan térben),
    dist: Distortion vagy None (ekkor G maga a torzított terű homográfia)."""

    def __init__(self, G, dist=None):
        self.G = np.asarray(G, np.float64)
        self.dist = dist

    def points(self, pts):
        p = cv2.perspectiveTransform(pts.astype(np.float64)[None], self.G)[0]
        return p if self.dist is None else self.dist.D(p)

    def grid(self, xs, ys):
        h = self.G
        z = h[2, 0] * xs + h[2, 1] * ys + h[2, 2]
        mx = ((h[0, 0] * xs + h[0, 1] * ys + h[0, 2]) / z).astype(np.float32)
        my = ((h[1, 0] * xs + h[1, 1] * ys + h[1, 2]) / z).astype(np.float32)
        if self.dist is not None:
            Dx, Dy, P = self.dist.dense_D()
            px, py = mx + P, my + P
            mx = cv2.remap(Dx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            my = cv2.remap(Dy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return mx, my

    def map_xy(self, xs, ys):
        """Leképezés tetszőleges (pl. durva rácsú) koordináta-tömbökre, float64 pontossággal."""
        h = self.G
        z = h[2, 0] * xs + h[2, 1] * ys + h[2, 2]
        mx = (h[0, 0] * xs + h[0, 1] * ys + h[0, 2]) / z
        my = (h[1, 0] * xs + h[1, 1] * ys + h[1, 2]) / z
        if self.dist is not None:
            if np.size(mx) > 4096:
                # nagy tömbre: az előre kiszámolt sűrű D-táblázatból bilineáris kiolvasás (sima mező,
                # 1 px-es rácson a hiba elhanyagolható), az iteratív inverz helyett
                Dx, Dy, P = self.dist.dense_D()
                px = (mx + P).astype(np.float32); py = (my + P).astype(np.float32)
                mx = cv2.remap(Dx, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
                my = cv2.remap(Dy, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
            else:
                mx, my = self.dist.D_xy(mx, my)
        return mx, my

    def inv_map_xy(self, x, y):
        """Fordított irány: képkocka (szenzor) pixel -> kimeneti vászon koordináta."""
        if self.dist is not None:
            dx, dy = self.dist._delta(x, y)
            x, y = x + dx, y + dy          # U: torzított -> torzítatlan
        Gi = np.linalg.inv(self.G)
        z = Gi[2, 0] * x + Gi[2, 1] * y + Gi[2, 2]
        return (Gi[0, 0] * x + Gi[0, 1] * y + Gi[0, 2]) / z, (Gi[1, 0] * x + Gi[1, 1] * y + Gi[1, 2]) / z

    def warp(self, img, W, H, coarse=None, interp=cv2.INTER_LINEAR, border=-1.0):
        """A képkocka a kimeneti vászonra (W x H). Torzítás nélkül pontos warpPerspective;
        torzítással a leképezés a durva rácson (coarse: CoarseGrid) számolódik és felskálázódik."""
        if self.dist is None:
            return cv2.warpPerspective(img, self.G, (W, H), flags=interp | cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        mx, my = coarse.up_map(*self.map_xy(coarse.xc, coarse.yc))
        return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=border)


class CoarseGrid:
    """Durva rács a kimeneti vásznon: a (j+0.5)*f-0.5 pixelközéppontok, ami pontosan megfelel a
    cv2.resize(INTER_LINEAR) felskálázásnak. A homográfia, a torzítás és az AP-eltolási mező is
    sima, így a durva rácson számolt, majd lineárisan felskálázott leképezés hibája elhanyagolható
    (<0.01 px), a számítás pedig f²-szer kevesebb."""

    def __init__(self, W, H, f):
        self.W, self.H, self.f = W, H, f
        self.ws, self.hs = -(-W // f), -(-H // f)
        xc = (np.arange(self.ws) + 0.5) * f - 0.5
        yc = (np.arange(self.hs) + 0.5) * f - 0.5
        self.yc, self.xc = np.meshgrid(yc, xc, indexing="ij")

    def up(self, a):
        big = cv2.resize(a.astype(np.float32), (self.ws * self.f, self.hs * self.f), interpolation=cv2.INTER_LINEAR)
        return big[:self.H, :self.W]

    def up_map(self, mx, my):
        return self.up(mx), self.up(my)


def calibrate(corr, W, H, deg=3, ridge=1e-4, max_pts=600, seed=0):
    """corr: lista (x_ref (N,2), x_k (N,2)) teljes felbontású px-ben, képkockánként.
    Visszaad: (Distortion | None, info dict)."""
    rng = np.random.default_rng(seed)
    data = []
    for xr, xk in corr:
        if len(xr) < 30:
            continue
        if len(xr) > max_pts:
            sel = rng.choice(len(xr), max_pts, replace=False)
            xr, xk = xr[sel], xk[sel]
        data.append((xr.astype(np.float64), xk.astype(np.float64)))
    if len(data) < 3:
        return None, dict(enabled=False, msg="kevés képkocka megfelelő pontpárokkal")
    M = len(_terms(deg))
    probe = Distortion(np.zeros(2 * M), deg, W, H)

    def resid(c):
        probe.coef = c.reshape(2, -1)
        out = []
        for xr, xk in data:
            ur = probe.U(xr); uk = probe.U(xk)
            G, _ = cv2.findHomography(ur, uk, 0)
            if G is None:
                out.append(np.full(2 * len(xr), 50.0)); continue
            pred = cv2.perspectiveTransform(ur[None], G)[0]
            out.append((pred - uk).ravel())
        out.append(np.sqrt(ridge * sum(len(d[0]) for d in data)) * probe.s * c)
        return np.concatenate(out)

    c0 = np.zeros(2 * M)
    r0 = resid(c0)[:-2 * M]
    sol = least_squares(resid, c0, loss="huber", f_scale=1.0, x_scale=1e-3, diff_step=1e-6, max_nfev=200)
    r1 = resid(sol.x)[:-2 * M]
    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))
    med = lambda r: float(np.median(np.hypot(r[0::2], r[1::2])))
    dist = Distortion(sol.x, deg, W, H)
    info = dict(enabled=True, frames=len(data), points=int(sum(len(d[0]) for d in data)),
                rms_before=rms(r0), rms_after=rms(r1), median_before=med(r0), median_after=med(r1),
                **dist.to_dict(), msg="ok")
    # csak akkor használjuk, ha érdemben javít
    if not (info["rms_after"] < 0.95 * info["rms_before"]):
        info.update(enabled=False, msg="a torzításmodell nem javított érdemben")
        return None, info
    return dist, info


def frame_homography(dist, H_dist, pts_ref):
    """A torzított terű H (ref->frame) átírása torzítatlan terű G-re a Hold mintapontjain."""
    xk = cv2.perspectiveTransform(pts_ref.astype(np.float64)[None], H_dist)[0]
    G, _ = cv2.findHomography(dist.U(pts_ref), dist.U(xk), 0)
    return G


def max_shift_in(dist, mask):
    """A torzítási korrekció legnagyobb értéke a maszkon (a Hold által bejárt területen) belül."""
    ys, xs = np.nonzero(mask[::16, ::16])
    if len(xs) == 0:
        return 0.0
    u = np.stack([xs * 16, ys * 16], 1).astype(np.float64)
    return float(np.hypot(*(dist.D(u) - u).T).max())


def draw_field(dist, W, H, mask=None, step=160, gain=20.0):
    """A torzítási mező nyilas ábrája (gain-szeres nagyítással), csak a maszkon (Hold-terület) belül;
    azon kívül a polinom csak extrapolál, ott nincs mérési adat."""
    img = np.full((H // 2, W // 2, 3), 255, np.uint8)
    if mask is not None:
        img[cv2.resize(mask.astype(np.uint8), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST) > 0] = (235, 245, 235)
    for y in range(step // 2, H, step):
        for x in range(step // 2, W, step):
            if mask is not None and not mask[y, x]:
                continue
            u = np.array([[x, y]], np.float64)
            d = dist.D(u)[0] - u[0]
            p0 = (int(x / 2), int(y / 2))
            p1 = (int((x + gain * d[0]) / 2), int((y + gain * d[1]) / 2))
            cv2.arrowedLine(img, p0, p1, (40, 40, 200), 2, tipLength=0.25)
    mx = max_shift_in(dist, mask) if mask is not None else dist.max_shift()
    cv2.putText(img, f"torzitas x{gain:.0f}, max a Hold teruleten {mx:.1f} px", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img
