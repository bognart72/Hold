"""Hold-maszk, élességmérés és globális (homográfiás) illesztés.

Konvenció: H (3x3) a REFERENCIA koordinátáit képezi a képkocka koordinátáira (ref -> frame),
ahogy a cv2.findTransformECC is. A referenciába vetítés: warpPerspective(..., WARP_INVERSE_MAP).
"""
import numpy as np
import cv2


def moon_mask(small, level=0.25):
    """Célpont-maszk kicsinyített szürke képen. A sötét égi háttér és az okulár-blende kimarad.
    level: a küszöb helye a háttér (2. percentilis) és a csúcsfény (99.7.) között.
    Hold: 0.25 (a terminátor menti sötétebb részek is bent maradjanak).
    Nap: 0.4 — a túlexponált korong körüli fényudvar (szórt fény) a Hold-küszöb fölé érhet,
    a korong pereme viszont éles ugrás, ennek közepére esik a 0.4-es küszöb."""
    b = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 2)
    hi = np.percentile(b, 99.7)
    lo = np.percentile(b, 2)
    m = (b > lo + level * (hi - lo)).astype(np.uint8)
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


def circle_mask(mask, iters=300, tol=1.5, seed=0):
    """A maszk peremére robusztus (RANSAC + legkisebb négyzetes) körillesztés, és a kör belseje
    mint új maszk. A Nap korongja kör: így a fényudvar dudorai, a tükröződések és a perem
    zajos kontúrja nem kerülnek a maszkba. Ha az illesztés nem megbízható, az eredeti maszk marad."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask
    P = max(cnts, key=len)[:, 0, :].astype(np.float64)
    H, W = mask.shape
    # a képkocka szélén levő kontúrpontok nem a perem részei (a korong kilóghat)
    P = P[(P[:, 0] > 1) & (P[:, 0] < W - 2) & (P[:, 1] > 1) & (P[:, 1] < H - 2)]
    if len(P) < 30:
        return mask
    rng = np.random.default_rng(seed)

    def fit(pts):
        A = np.c_[2 * pts, np.ones(len(pts))]
        b = (pts ** 2).sum(1)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        r = np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 0))
        return c[0], c[1], r

    best, best_in = None, None
    for _ in range(iters):
        cx, cy, r = fit(P[rng.choice(len(P), 3, replace=False)])
        d = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)
        inl = d < tol
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (cx, cy, r), inl
    if best_in.sum() < 0.5 * len(P):
        return mask
    cx, cy, r = fit(P[best_in])
    yy, xx = np.mgrid[0:H, 0:W]
    return np.hypot(xx - cx, yy - cy) <= r - 0.5


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


def limb_sharpness(gray, band):
    """A korong peremének élessége: a gradiens felső 30%-ának átlaga a peremsávban / fényesség.
    A seeing elmossa a peremet, így ez a Napon megbízható élességmérték akkor is, ha a felszínen
    alig van részlet (a felszíni Laplace-mérték ott főleg zajt mérne)."""
    if not band.any():
        return 0.0
    gb = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gm = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3))
    v = gm[band]
    top = v[v >= np.percentile(v, 70)]
    ref = float(np.percentile(gb[band], 90)) + 1e-6
    return float(top.mean()) / ref


def disc_params(mask):
    """Korong középpontja (x, y) és egyenértékű sugara egy maszkból."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean()), float(np.sqrt(len(xs) / np.pi))


def scale_h(Hs, ds):
    """Kicsinyített (1/ds) koordinátákban adott H -> teljes felbontás."""
    c = (ds - 1) / 2.0
    S = np.array([[ds, 0, c], [0, ds, c], [0, 0, 1]], np.float64)
    return S @ Hs @ np.linalg.inv(S)


def _u8(x):
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def dilate(mask, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def align_one(ref_small, ref_mask, ref_kp, ref_desc, small, mask, sun=False, spots_ref=None, spots_frm=None,
              ds=4):
    """Egy képkocka illesztése a referenciához: AKAZE+RANSAC durva becslés, majd ECC.

    Hold: ECC homográfia a korong belsején (sok struktúra).
    Nap (sun=True): kevés felszíni struktúra → a durva becslés a korong középpontjából és
    sugarából jön (ha a jellemzőpontok kevesek), az ECC AFFIN modellel fut, és a maszk a perem
    sávját is tartalmazza (a perem adja a legerősebb, legmegbízhatóbb illesztési jelet).

    Visszaad: (H ref->frame kicsinyített koordinátákban, ecc korreláció, módszer)."""
    s = small.astype(np.float32)
    m = dilate(mask, 6) if sun else erode(mask, 4)
    if sun:
        return _align_sun(ref_small, ref_mask, s, mask, m, spots_ref, spots_frm, ds)
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
    if H0 is None and sun:
        dr, df = disc_params(ref_mask), disc_params(mask)
        if dr and df:
            method = "disc"
            sc = df[2] / dr[2]
            H0 = np.array([[sc, 0, df[0] - sc * dr[0]], [0, sc, df[1] - sc * dr[1]], [0, 0, 1]], np.float64)
    if H0 is None:
        method = "phase"
        win = cv2.createHanningWindow(s.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(ref_small * ref_mask, s * m, win)
        H0 = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    motion = cv2.MOTION_AFFINE if sun else cv2.MOTION_HOMOGRAPHY

    def ecc(tpl, img, Hinit, crit, msk, gfs):
        w0 = (Hinit[:2] if motion == cv2.MOTION_AFFINE else Hinit).astype(np.float32)
        cc_, w = cv2.findTransformECC(tpl, img, w0, motion, crit, msk, gfs)
        w = w.astype(np.float64)
        return cc_, (np.vstack([w, [0, 0, 1]]) if motion == cv2.MOTION_AFFINE else w)

    try:
        # két szintű piramis: előbb fél felbontáson, aztán a kicsinyített képen
        S = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1]])
        Si = np.linalg.inv(S)
        r2 = cv2.resize(ref_small, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(s, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        m2 = cv2.resize(m.astype(np.uint8), (s2.shape[1], s2.shape[0]), interpolation=cv2.INTER_NEAREST)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6)
        _, H2 = ecc(r2, s2, S @ H0 @ Si, crit, m2, 3)
        H1 = Si @ H2 @ S
        # finom szint teljes konvergenciáig (minőség az első: a kisebb globális maradék a
        # helyi AP-mérésnek is jobb kiindulás; 12 iterációval mérhetően nőtt a maradék)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-6)
        cc, Hw = ecc(ref_small, s, H1, crit, m.astype(np.uint8), 5)
        # józansági ellenőrzés: az ECC ne vigye messzire a durva becsléstől
        c = np.array([[[s.shape[1] / 2, s.shape[0] / 2]]], np.float32)
        d = np.linalg.norm(cv2.perspectiveTransform(c, Hw) - cv2.perspectiveTransform(c, H0))
        if d > 0.05 * max(s.shape):
            return H0, float("nan"), method + "+ecc_rejected"
        return Hw, float(cc), method + "+ecc"
    except cv2.error:
        return H0, float("nan"), method + "+ecc_failed"


def find_spots(gray, inner, min_area=3, max_area=5000, thr=0.88):
    """Sötét foltok (napfoltok) keresése: a lokális háttérhez képest legalább 12%-kal sötétebb,
    kompakt területek a korong belsejében. Visszaad: lista (x, y, terület, mélység), a hely a
    sötétséggel súlyozott középpont (szubpixel pontos)."""
    gb = cv2.GaussianBlur(gray, (0, 0), 1.2)
    w = inner.astype(np.float32)
    bg = cv2.GaussianBlur(gb * w, (0, 0), 12) / np.maximum(cv2.GaussianBlur(w, (0, 0), 12), 1e-3)
    rel = gb / np.maximum(bg, 1e-4)
    dark = ((rel < thr) & inner).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    out = []
    for j in range(1, n):
        a = int(st[j, cv2.CC_STAT_AREA])
        if not (min_area <= a <= max_area):
            continue
        x0, y0, ww, hh = st[j, 0], st[j, 1], st[j, 2], st[j, 3]
        sub = (lab[y0:y0 + hh, x0:x0 + ww] == j)
        wt = (1 - rel[y0:y0 + hh, x0:x0 + ww]) * sub
        yy, xx = np.mgrid[y0:y0 + hh, x0:x0 + ww]
        sw = wt.sum() + 1e-9
        out.append((float((xx * wt).sum() / sw), float((yy * wt).sum() / sw), a, float(1 - rel[lab == j].min())))
    return out


def _align_sun_spots(H0, spots_ref, spots_frm, ds, r_small):
    """A korongból kapott kezdő becslést (H0, kicsinyített koordinátákban) a napfoltok párosításával
    finomítja. ≥2 egyező folt: hasonlósági transzformáció (eltolás + forgatás + méretarány, RANSAC),
    1 folt: csak eltolás. Hamis forgatás ellen: legfeljebb 3°, a méretarány a korongétól ≤3%-kal térhet el."""
    if not spots_ref or not spots_frm:
        return None, None
    c = (ds - 1) / 2.0
    R = np.array([[(x - c) / ds, (y - c) / ds] for x, y, a, d in spots_ref], np.float64)
    F = np.array([[(x - c) / ds, (y - c) / ds] for x, y, a, d in spots_frm], np.float64)
    ar = np.array([a for *_, a, d in spots_ref], np.float64)
    af = np.array([a for *_, a, d in spots_frm], np.float64)
    pred = (H0[:2, :2] @ R.T).T + H0[:2, 2]
    tol = 0.15 * r_small   # a korong-becslés és a foltok eltérése hosszú videón ~40 px is lehet
    pr, pf = [], []
    used = set()
    for k in np.argsort(-ar):                 # a nagyobb foltok előbb
        d = np.hypot(*(F - pred[k]).T)
        j = int(np.argmin(d))
        if d[j] < tol and j not in used and 0.33 <= af[j] / ar[k] <= 3.0:
            used.add(j); pr.append(R[k]); pf.append(F[j])
    if not pr:
        return None, None
    pr, pf = np.array(pr), np.array(pf)
    sc0 = np.hypot(H0[0, 0], H0[1, 0])
    if len(pr) >= 2:
        A, inl = cv2.estimateAffinePartial2D(pr.astype(np.float32), pf.astype(np.float32), method=cv2.RANSAC,
                                             ransacReprojThreshold=1.0)
        if A is not None and inl is not None and inl.sum() >= 2:
            sc = np.hypot(A[0, 0], A[1, 0]); rot = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
            if abs(rot) <= 3.0 and abs(sc / sc0 - 1) <= 0.03:
                return np.vstack([A, [0, 0, 1]]).astype(np.float64), f"sun_spots_sim{int(inl.sum())}"
    # egy (vagy nem konzisztens több) folt: csak eltolás-korrekció a legnagyobb egyező folt szerint
    Hc = H0.copy()
    dlt = pf[0] - ((H0[:2, :2] @ pr[0]) + H0[:2, 2])
    Hc[0, 2] += dlt[0]; Hc[1, 2] += dlt[1]
    return Hc, "sun_spots_t"


def _align_sun(ref_small, ref_mask, s, mask, m, spots_ref=None, spots_frm=None, ds=4):
    """Nap: globális illesztés FORGATÁS NÉLKÜL.

    A korong kör, így önmagában nem határozza meg az elforgatást; a beégett, szinte egyenletes
    felszínen a néhány pici folt kevés ehhez. Szabad forgatással az ECC „elcsúszik” (valódi
    felvételen 0…16° hamis forgatás → a foltok megkettőződtek). A valódi mezőrotáció 20–30 s alatt
    < 0,2° (a peremen 1–2 px), ezt a helyi AP-mérés korrigálja. Ezért: méretarány és középpont a
    korongra illesztett körből, majd ECC csak eltolással (a perem sávja pontosan rögzíti)."""
    dr, df = disc_params(ref_mask), disc_params(mask)
    if not (dr and df):
        return np.eye(3), float("nan"), "sun_disc_failed"
    sc = df[2] / dr[2]
    H0 = np.array([[sc, 0, df[0] - sc * dr[0]], [0, sc, df[1] - sc * dr[1]], [0, 0, 1]], np.float64)
    # elsődleges: a napfoltokra illesztés (a foltok egyezése a lényeg; a perem és a belső rész a
    # látómező torzítása miatt eltérően mozoghat – valódi 146 s-os videón ~20 px)
    Hs, how = _align_sun_spots(H0, spots_ref, spots_frm, ds, dr[2])
    if Hs is not None:
        return Hs, float("nan"), how
    # tartalék: a méretarányt a kör adja; az eltolást ECC finomítja a méretarány-korrigált referencián
    hh, ww = s.shape
    ref_sc = cv2.warpAffine(ref_small, H0[:2], (ww, hh), flags=cv2.INTER_LINEAR)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        cc, T = cv2.findTransformECC(ref_sc, s, np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION,
                                     crit, m.astype(np.uint8), 5)
        tx, ty = float(T[0, 2]), float(T[1, 2])
        if np.hypot(tx, ty) > 0.03 * max(hh, ww):
            return H0, float("nan"), "sun_disc+ecc_rejected"
        Hf = H0.copy(); Hf[0, 2] += tx; Hf[1, 2] += ty
        return Hf, float(cc), "sun_disc+ecc_t"
    except cv2.error:
        return H0, float("nan"), "sun_disc"


def reference_features(ref_small, ref_mask):
    akaze = cv2.AKAZE_create(threshold=0.0005)
    return akaze.detectAndCompute(_u8(ref_small), erode(ref_mask, 4).astype(np.uint8) * 255)
