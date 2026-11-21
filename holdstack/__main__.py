"""Parancssor: python -m holdstack <videó vagy képmappa> -o <kimeneti mappa> [opciók]

--auto: előbb a felvételhez hangolja a paramétereket (holdstack.autotune), majd azokkal fut.
A kifejezetten megadott opciók (pl. --percent 25) a hangolt értéket is felülírják.
"""
import argparse

from .pipeline import Config, run
from .autotune import DEFAULTS


def main():
    p = argparse.ArgumentParser(prog="holdstack", description="Hold-stackelő (lucky imaging, multi-AP, önkalibráló flat)")
    p.add_argument("input", help="videófájl vagy képeket tartalmazó mappa")
    p.add_argument("-o", "--out", default="eredmeny", help="kimeneti mappa")
    p.add_argument("--target", default="moon", choices=["moon", "hold", "sun", "nap"],
                   help="célpont: moon/hold (alap) vagy sun/nap (fehér fényű napszűrős felvétel)")
    p.add_argument("--dust", dest="dust", action="store_true", default=None,
                   help="szenzorhoz rögzített porszemek eltávolítása (Napnál alapból be)")
    p.add_argument("--no-dust", dest="dust", action="store_false", help="porszem-eltávolítás kikapcsolása")
    p.add_argument("--no-sun-enhance", action="store_true", help="Nap: a kiemelt (_nap_kiemelt) változat elhagyása")
    p.add_argument("--no-sun-detail", action="store_true",
                   help="Nap: a részlet-változat (_nap_reszlet: umbra–penumbra + fáklyamezők) elhagyása")
    p.add_argument("--facula", type=float, default=2.5,
                   help="Nap: fáklyamező-kiemelés erőssége (alap: 2.5; 0 = nincs)")
    p.add_argument("--auto",action="store_true", help="paraméterek hangolása ehhez a felvételhez a futás előtt")
    p.add_argument("--trial", action="store_true",
                   help="próbafuttatásos kalibráció (AP-box × arány mini-stackekkel), majd teljes futás a legjobbal")
    p.add_argument("--trial-only", action="store_true",
                   help="csak a próbafuttatásos kalibráció; az eredmény <név>_kalibracio.json, stack nem készül")
    # a hangolható opciók alapértéke None: így tudjuk, mit adott meg kifejezetten a felhasználó
    p.add_argument("--percent", type=float, help=f"AP-nként a legjobb %% (alap: {DEFAULTS['percent']})")
    p.add_argument("--min-frames", type=int, help=f"legalább ennyi kocka (alap: {DEFAULTS['min_frames']})")
    p.add_argument("--candidates", type=float, help=f"videónál a globálisan legjobb %% (alap: {DEFAULTS['candidates']})")
    p.add_argument("--half-box", type=int, help=f"AP-box félszélesség px-ben (alap: {DEFAULTS['half_box']})")
    p.add_argument("--step", type=int, default=0, help="AP-rácsosztás (0 = 1.5 * half-box)")
    p.add_argument("--search", type=int, help=f"lokális keresési sugár px-ben (alap: {DEFAULTS['search']})")
    p.add_argument("--no-refine", action="store_true", help="második (finomító) mérés kihagyása")
    p.add_argument("--no-flat", action="store_true", help="vinyettálás-korrekció kikapcsolása")
    p.add_argument("--flat-deg", type=int, default=3)
    p.add_argument("--no-gain", action="store_true")
    p.add_argument("--sharpen", type=str, help=f"wavelet rétegerősségek (alap: {DEFAULTS['sharpen']})")
    p.add_argument("--no-distortion", action="store_true", help="torzítás-önkalibráció kikapcsolása")
    p.add_argument("--dist-deg", type=int, default=3, help="torzításpolinom foka (2-4)")
    p.add_argument("--stretch", action="store_true",
                   help="automatikus széthúzás fej-térrel (alapból ki: a fényesség = a bemenetével)")
    p.add_argument("--knee", type=float, help=f"lágy csúcsfény-levágás kezdete (alap: {DEFAULTS['knee']})")
    p.add_argument("--no-ca", action="store_true", help="kromatikus aberráció korrekciójának kikapcsolása")
    p.add_argument("--defringe", type=float, help=f"színszegély-csillapítás 0-1 (alap: {DEFAULTS['defringe']})")
    p.add_argument("--mineral", type=float, default=0.0,
                   help="'mineral moon' kép készítése ezzel a színerősítéssel (pl. 4; 0 = nem készül)")
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    from .sysutil import disable_power_throttling
    disable_power_throttling()

    v = dict(DEFAULTS)
    if a.auto:
        from .autotune import analyse, suggest
        print("[  hangolás ] a felvétel előzetes elemzése…", flush=True)
        tuned, why = suggest(analyse(a.input))
        v.update(tuned)
        print("[  hangolás ] javasolt beállítások:", flush=True)
        for w in why:
            print("    - " + w, flush=True)
    for key, arg in (("percent", a.percent), ("min_frames", a.min_frames), ("candidates", a.candidates),
                     ("half_box", a.half_box), ("search", a.search), ("knee", a.knee),
                     ("defringe", a.defringe), ("sharpen", a.sharpen)):
        if arg is not None:
            v[key] = arg
    cfg = Config(stack_percent=float(v["percent"]), min_frames=int(v["min_frames"]),
                 candidate_percent=float(v["candidates"]), half_box=int(v["half_box"]), step=a.step,
                 search=int(v["search"]), refine=not a.no_refine,
                 flat=v["flat"] and not a.no_flat, flat_deg=a.flat_deg, gain=not a.no_gain,
                 distortion=v["distortion"] and not a.no_distortion, dist_deg=a.dist_deg,
                 stretch=a.stretch, knee=float(v["knee"]), ca=not a.no_ca, defringe=float(v["defringe"]),
                 mineral=a.mineral, sharpen=tuple(float(x) for x in str(v["sharpen"]).split(",")),
                 trial=a.trial, trial_only=a.trial_only, workers=a.workers,
                 target="sun" if a.target in ("sun", "nap") else "moon", dust=a.dust,
                 sun_enhance=not a.no_sun_enhance, sun_detail=not a.no_sun_detail, facula_gain=a.facula)
    run(a.input, a.out, cfg)


if __name__ == "__main__":
    main()
