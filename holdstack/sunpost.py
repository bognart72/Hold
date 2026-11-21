"""Nap-utófeldolgozás: a napfoltok umbra–penumbra szerkezetének visszaadása és a fáklyamezők kiemelése.

1. PSF a peremből: a napkorong pereme éles lépcső; a stacken látszó elmosódottsága a teljes
   (légköri + optikai + stack) elmosódás. A sugárirányú peremprofil deriváltjából (vonal-szórásfüggvény)
   Gauss-PSF szigmát becslünk.
2. Richardson–Lucy dekonvolúció ezzel a PSF-fel, csillapítással (az arányt korlátozzuk, ez a
   zajerősítést és a gyűrűzést fékezi), a perem közelében fokozatos visszavezetéssel az eredetibe.
3. Foltszerkezet: a foltok környezetében (ahol a kép a helyi háttérnél sötétebb) finom, kis léptékű
   kontrasztkiemelés, hogy az umbra–penumbra határ és a penumbra rostjai jobban elváljanak.
4. Fáklyamezők: a szélsötétedés kiegyenlítése után sávszűrt (≈2–25 px) POZITÍV eltérés erősítése,
   a perem felé növekvő súllyal (a fáklyamezők a perem közelében látszanak).
A színarányt minden lépés megtartja (csak a fényességet módosítjuk).
"""
import numpy as np
import cv2


def _lum(img):
    return img if img.ndim == 2 else (0.0722 * img[..., 0] + 0.7152 * img[..., 1] + 0.2126 * img[..., 2])


def disc(mask):
    ys, xs = np.nonzero(mask)
    return xs.mean(), ys.mean(), np.sqrt(len(xs) / np.pi)


def limb_psf_sigma(L, mask, n_ang=720):
    """Gauss-PSF szigma (px) a perem profiljára illesztett elmosott lépcsőből (erf). None, ha nem mérhető."""
    from .sharpen import limb_psf_sigma as lps
    return lps(L, mask, n_ang)


def _limb_psf_sigma_moment(L, mask, n_ang=720):
    """RÉGI becslés (második momentum ±3 px-es ablakban) — kb. 1,6 px fölött telítődik, nem használt."""
    cx, cy, R = disc(mask)
    rr = np.arange(-20, 20.01, 0.25)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    xs = (cx + (R + rr[None, :]) * np.cos(ang[:, None])).astype(np.float32)
    ys = (cy + (R + rr[None, :]) * np.sin(ang[:, None])).astype(np.float32)
    prof = cv2.remap(L.astype(np.float32), xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # minden szögnél külön a profil legmeredekebb pontjára igazítva (a perem nem tökéletes kör)
    d = -np.gradient(prof, rr, axis=1)
    sig = []
    for k in range(n_ang):
        dk = d[k]
        i = int(np.argmax(dk))
        if i < 12 or i > len(rr) - 13:
            continue
        seg = dk[i - 12:i + 13].copy()
        seg -= np.median(dk)                 # háttér-gradiens (szélsötétedés) le
        seg = np.clip(seg, 0, None)
        if seg.sum() <= 0:
            continue
        x = rr[i - 12:i + 13]
        mu = (seg * x).sum() / seg.sum()
        sig.append(np.sqrt((seg * (x - mu) ** 2).sum() / seg.sum()))
    if len(sig) < n_ang * 0.3:
        return None
    s = float(np.median(sig))
    return s if 0.4 <= s <= 8 else None


def rl_deconv(L, sigma, iters=25, clip=1.6, mask=None):
    """Richardson–Lucy dekonvolúció Gauss-PSF-fel, csillapítással (az arány [1/clip, clip]-re vágva)."""
    L = np.clip(L.astype(np.float32), 1e-5, None)
    u = L.copy()
    for _ in range(iters):
        est = cv2.GaussianBlur(u, (0, 0), sigma)
        ratio = np.clip(L / np.maximum(est, 1e-5), 1 / clip, clip)
        u = u * cv2.GaussianBlur(ratio, (0, 0), sigma)
    return u


def spot_mask(L, mask, rho, thr=0.90, max_frac=0.01):
    """Napfoltok maszkja: a helyi (foltmentes) háttérnél thr-nél sötétebb, kompakt területek a korong
    belsejében (ρ < 0.95). A korong területének max_frac-ánál nagyobb összefüggő terület nem folt."""
    inner = mask & (rho < 0.95)
    w = inner.astype(np.float32)
    gb = cv2.GaussianBlur(L, (0, 0), 1.0)
    bg = cv2.GaussianBlur(gb * w, (0, 0), 15) / np.maximum(cv2.GaussianBlur(w, (0, 0), 15), 1e-3)
    # második kör: a háttérből a sötét pixelek kizárva (a nagy folt ne húzza le a saját hátterét)
    q = ((gb > 0.95 * bg) & inner).astype(np.float32)
    bg = cv2.GaussianBlur(gb * q, (0, 0), 20) / np.maximum(cv2.GaussianBlur(q, (0, 0), 20), 1e-3)
    dark = ((gb < thr * np.maximum(bg, 1e-4)) & inner).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    amax = max_frac * float(mask.sum())
    keep = np.zeros(L.shape, bool)
    for j in range(1, n):
        if 3 <= st[j, cv2.CC_STAT_AREA] <= amax:
            keep |= lab == j
    return keep


def enhance(img, mask, sigma=None, iters=30, spot_gain=0.0, facula_gain=2.5, limb_fade=0.035, psf_frac=0.7,
            surface_deconv=0.0, log=None):
    """surface_deconv: a dekonvolúció aránya a foltokon KÍVÜL (0 = csak a foltoknál, 1 = mindenhol).
    A foltoknál teljes; a sima felszínen a dekonvolúció főleg zajt erősítene (szintetikus mérés:
    a granuláció-egyezés 0,90 → 0,79 teljes dekonvolúcióval)."""
    """Visszaad: (kiemelt kép [a bemenettel azonos skálán], info dict)."""
    im = np.clip(img.astype(np.float32), 0, None)
    L = _lum(im)
    cx, cy, R = disc(mask)
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    rho = np.hypot(xx - cx, yy - cy) / R
    info = {}
    if sigma is None:
        sigma = limb_psf_sigma(L, mask)
    info["psf_sigma"] = sigma
    # 1–2. dekonvolúció (csak ha a PSF megbízhatóan mérhető)
    if sigma is not None:
        Lrl = rl_deconv(L, sigma * psf_frac, iters)    # alábecsült PSF: kevesebb túllövés
        # hol van folt: kompakt, a helyi háttérnél ≥10%-kal sötétebb területek (a foltkereső szabálya);
        # a nagy kiterjedésű, enyhe sötétedés (szélsötétedés, felszíni foltosság) NEM folt
        dark = spot_mask(L, mask, rho)
        near = cv2.dilate(dark.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(6 * sigma) * 2 + 1,) * 2))
        wspot = cv2.GaussianBlur(near.astype(np.float32), (0, 0), 2 * sigma)
        wdec = surface_deconv + (1 - surface_deconv) * np.clip(wspot, 0, 1)
        w = np.clip((1 - limb_fade - rho) / limb_fade, 0, 1) * wdec   # a peremnél vissza az eredetibe
        Ld = w * Lrl + (1 - w) * L
    else:
        Ld = L.copy()
    # helyi háttér a foltok nélkül (normált konvolúció)
    m = mask.astype(np.float32)
    pre = cv2.GaussianBlur(Ld * m, (0, 0), 25) / np.maximum(cv2.GaussianBlur(m, (0, 0), 25), 1e-3)
    quiet = (Ld > 0.9 * pre).astype(np.float32) * m
    bg = cv2.GaussianBlur(Ld * quiet, (0, 0), 25) / np.maximum(cv2.GaussianBlur(quiet, (0, 0), 25), 1e-3)
    bg = np.maximum(bg, 1e-4)
    r = Ld / bg                                   # relatív fényesség: 1 = nyugodt fotoszféra
    # 3. foltszerkezet: kis léptékű kontraszt erősítése ott, ahol folt van (r < ~0.95)
    spotw = np.clip((0.97 - cv2.GaussianBlur(r, (0, 0), 3)) / 0.15, 0, 1)
    spotw = cv2.GaussianBlur(spotw, (0, 0), 3)
    detail = r - cv2.GaussianBlur(r, (0, 0), 2.0)
    r_s = r + spot_gain * spotw * detail
    # 4. fáklyamezők: sávszűrt pozitív eltérés, a perem felé erősödő súllyal
    # normált konvolúció a korongon belül: a perem külső sötétje ne hozzon létre hamis világos gyűrűt
    def nblur(x, s):
        return cv2.GaussianBlur(x * m, (0, 0), s) / np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)
    band = nblur(r, 2.0) - nblur(r, 25.0)
    fac = np.clip(band, 0, None) * (1 - spotw)
    # a legszélső sávban (a perem elmosódott lépcsője) nincs kiemelés
    wlimb = (0.3 + 0.7 * np.clip(rho, 0, 1) ** 2) * np.clip((0.985 - rho) / 0.015, 0, 1)
    r_f = r_s + facula_gain * wlimb * fac
    Lout = np.where(mask, r_f * bg, Ld)
    ratio = Lout / np.maximum(L, 1e-5)
    out = im * ratio[..., None] if im.ndim == 3 else Lout
    info["spot_area_px"] = int(spot_mask(L, mask, rho).sum())
    return np.clip(out, 0, None), info
