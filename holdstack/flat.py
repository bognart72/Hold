"""Önkalibráló flat: a szenzorhoz rögzített vinyettálás becslése magából a felvételből.

Modell (log térben): log I_k(x) = log g(x) + log R(p) , ahol x = H_k(p) a szenzor-koordináta,
p a Hold-koordináta (referencia), R a Hold albedója, g a vinyettálás.
Mivel a Hold vándorol a szenzoron, ugyanaz a p több x helyen is látszik, így g becsülhető.
g egy alacsony fokú 2D polinom (konstans tag nélkül: g(képközép) = 1).

Megjegyzés: ha a képkockánkénti erősítés is szabad lenne, a polinom lineáris tagja
nem lenne azonosítható (egyenletes drift esetén). Ezért a modellt rögzített expozícióval
illesztjük, és a képkocka-erősítést utólag, a maradékból becsüljük.
"""
import numpy as np
import cv2

from .geometry import erode


def _terms(deg):
    return [(i, j) for d in range(1, deg + 1) for i in range(d + 1) for j in [d - i]]


def _basis(u, v, deg):
    return np.stack([u ** i * v ** j for i, j in _terms(deg)], axis=-1)


def norm_coords(x, y, W, H):
    return (x + 0.5) / W * 2 - 1, (y + 0.5) / H * 2 - 1


def calibrate(smalls, masks, Hs_small, frames, ref_i, deg=3, nsamp=5000, seed=0):
    """Visszaad: dict(coef, deg, gains{frame: log-erősítés}, rms, spread, enabled, msg)."""
    Hh, Ww = smalls[ref_i].shape
    rng = np.random.default_rng(seed)
    rm = erode(masks[ref_i], 12)
    ys, xs = np.nonzero(rm)
    if len(xs) < 100:
        return dict(enabled=False, msg="túl kicsi Hold-maszk")
    sel = rng.choice(len(xs), size=min(nsamp, len(xs)), replace=False)
    pts = np.float32(np.stack([xs[sel], ys[sel]], 1))[None]
    J, K, U, V, Y = [], [], [], [], []
    centers = []
    for k in frames:
        b = cv2.GaussianBlur(smalls[k].astype(np.float32), (0, 0), 2)
        p = cv2.perspectiveTransform(pts, Hs_small[k])[0]
        fm = erode(masks[k], 8)
        xi = np.round(p[:, 0]).astype(int)
        yi = np.round(p[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < Ww) & (yi >= 0) & (yi < Hh)
        ok[ok] &= fm[yi[ok], xi[ok]]
        val = cv2.remap(b, p[:, 0].reshape(1, -1).astype(np.float32), p[:, 1].reshape(1, -1).astype(np.float32),
                        cv2.INTER_LINEAR)[0]
        ok &= val > 0.02
        idx = np.nonzero(ok)[0]
        u, v = norm_coords(p[idx, 0], p[idx, 1], Ww, Hh)
        J.append(idx); K.append(np.full(len(idx), k)); U.append(u); V.append(v); Y.append(np.log(val[idx]))
        c = cv2.perspectiveTransform(np.float32([[[Ww / 2, Hh / 2]]]), Hs_small[k])[0, 0]
        centers.append(norm_coords(c[0], c[1], Ww, Hh))
    J = np.concatenate(J); K = np.concatenate(K); U = np.concatenate(U); V = np.concatenate(V); Y = np.concatenate(Y)
    centers = np.array(centers)
    spread = float(np.hypot(*centers.std(0)))
    if spread < 0.02:
        return dict(enabled=False, spread=spread, msg="a Hold alig mozdult a szenzoron, a flat nem becsülhető")
    B = _basis(U, V, deg)
    nS = pts.shape[1]
    w = np.ones(len(Y))
    c = np.zeros(B.shape[1])
    lam = 1e-3 * len(Y)
    for _ in range(12):
        # albedó minden mintapontra: súlyozott átlag
        e = Y - B @ c
        r = np.bincount(J, w * e, nS) / np.maximum(np.bincount(J, w, nS), 1e-9)
        t = Y - r[J]
        sw = np.sqrt(w)[:, None]
        A = B * sw
        c = np.linalg.solve(A.T @ A + lam * np.eye(B.shape[1]), A.T @ (t * sw[:, 0]))
        res = Y - B @ c - r[J]
        mad = np.median(np.abs(res - np.median(res))) * 1.4826 + 1e-6
        w = np.minimum(1.0, 1.5 * mad / (np.abs(res) + 1e-12))  # Huber-súly
    res = Y - B @ c - r[J]
    gains = {int(k): float(np.median(res[K == k])) for k in frames if np.any(K == k)}
    # a flat terjedelme a ténylegesen bejárt területen
    gv = np.exp(B @ c)
    return dict(enabled=True, coef=c.tolist(), deg=deg, gains=gains, rms=float(np.sqrt(np.mean(res ** 2))),
                spread=spread, g_min=float(gv.min()), g_max=float(gv.max()), msg="ok")


def flat_field(calib, W, H):
    """A becsült vinyettálás teljes felbontásban (float32), [0.3, 3] közé vágva."""
    if not calib.get("enabled"):
        return None
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    u, v = norm_coords(x[None, :], y[:, None], W, H)
    u = np.broadcast_to(u, (H, W)); v = np.broadcast_to(v, (H, W))
    logg = np.zeros((H, W), np.float32)
    for (i, j), cc in zip(_terms(calib["deg"]), calib["coef"]):
        logg += np.float32(cc) * (u ** i) * (v ** j)
    return np.clip(np.exp(logg), 0.3, 3.0).astype(np.float32)
