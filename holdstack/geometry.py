"""Hold-maszk, élességmérés és globális (homográfiás) illesztés.

Konvenció: H (3x3) a REFERENCIA koordinátáit képezi a képkocka koordinátáira (ref -> frame),
ahogy a cv2.findTransformECC is. A referenciába vetítés: warpPerspective(..., WARP_INVERSE_MAP).
"""
import numpy as np
import cv2


def moon_mask(small):
    """Hold-maszk kicsinyített szürke képen. A sötét égi háttér és az okulár-blende kimarad."""
    b = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 2)
    hi = np.percentile(b, 99.7)
    lo = np.percentile(b, 2)
    m = (b > lo + 0.25 * (hi - lo)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros_like(m, bool)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == k).astype(np.uint8)
    # lyukak kitöltése (sötét kráterbelsők, terminátor-árnyékok)
    inv = 1 - m
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv)
    H, W = m.shape
    for j in range(1, n2):
        x, y, w, h, _a = st2[j]
        if x > 0 and y > 0 and x + w < W and y + h < H:
            m[lab2 == j] = 1
    return m.astype(bool)


def erode(mask, r):
    if r <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def sharpness(gray, mask, sigma=1.5):
    """Zajtűrő élesség: Gauss utáni Laplace-variancia / átlagfényesség² a maszkon belül."""
    gb = cv2.GaussianBlur(gray, (0, 0), sigma)
    lap = cv2.Laplacian(gb, cv2.CV_32F)
    if not mask.any():
        return 0.0
    mu = float(gb[mask].mean())
    return float(lap[mask].var() / (mu * mu + 1e-12))


def scale_h(Hs, ds):
    """Kicsinyített (1/ds) koordinátákban adott H -> teljes felbontás."""
    c = (ds - 1) / 2.0
    S = np.array([[ds, 0, c], [0, ds, c], [0, 0, 1]], np.float64)
    return S @ Hs @ np.linalg.inv(S)


def _u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def align_one(ref_small, ref_mask, ref_kp, ref_desc, small, mask):
    """Egy képkocka illesztése a referenciához: AKAZE+RANSAC durva becslés, majd ECC homográfia.

    Visszaad: (H ref->frame kicsinyített koordinátákban, ecc korreláció, módszer)."""
    s = small.astype(np.float32)
    m = erode(mask, 4)
    H0, method = None, "akaze"
    akaze = cv2.AKAZE_create(threshold=0.0005)
    kp, desc = akaze.detectAndCompute(_u8(s), m.astype(np.uint8) * 255)
    if desc is not None and ref_desc is not None and len(kp) >= 10:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = bf.knnMatch(ref_desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) >= 10:
            P1 = np.float32([ref_kp[g.queryIdx].pt for g in good])
            P2 = np.float32([kp[g.trainIdx].pt for g in good])
            A, inl = cv2.estimateAffinePartial2D(P1, P2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if A is not None and inl is not None and inl.sum() >= 8:
                H0 = np.vstack([A, [0, 0, 1]])
    if H0 is None:
        method = "phase"
        win = cv2.createHanningWindow(s.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(ref_small * ref_mask, s * m, win)
        H0 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    try:
        # két szintű piramis: előbb fél felbontáson, aztán a kicsinyített képen
        S = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]])
        Si = np.linalg.inv(S)
        r2 = cv2.resize(ref_small, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(s, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(m.astype(np.uint8), (s2.shape[1], s2.shape[0]), interpolation=cv2.INTER_NEAREST)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        _, H2 = cv2.findTransformECC(r2, s2, (S @ H0 @ Si).astype(np.float32), cv2.MOTION_HOMOGRAPHY,
                                     crit, m2, 3)
        H1 = Si @ H2.astype(np.float64) @ S
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-6)
        cc, Hw = cv2.findTransformECC(ref_small, s, H1.astype(np.float32), cv2.MOTION_HOMOGRAPHY,
                                      crit, m.astype(np.uint8), 5)
        Hw = Hw.astype(np.float64)
        # józansági ellenőrzés: az ECC ne vigye messzire a durva becsléstől
        c = np.array([[[s.shape[1] / 2, s.shape[0] / 2]]], np.float32)
        d = np.linalg.norm(cv2.perspectiveTransform(c, Hw) - cv2.perspectiveTransform(c, H0))
        if d > 0.05 * max(s.shape):
            return H0, float("nan"), method + "+ecc_rejected"
        return Hw, float(cc), method + "+ecc"
    except cv2.error:
        return H0, float("nan"), method + "+ecc_failed"


def reference_features(ref_small, ref_mask):
    akaze = cv2.AKAZE_create(threshold=0.0005)
    return akaze.detectAndCompute(_u8(ref_small), erode(ref_mask, 4).astype(np.uint8) * 255)
