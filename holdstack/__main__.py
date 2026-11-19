"""Parancssor: python -m holdstack <videó vagy képmappa> -o <kimeneti mappa> [opciók]"""
import argparse

from .pipeline import Config, run


def main():
    p = argparse.ArgumentParser(prog="holdstack", description="Hold-stackelő (lucky imaging, multi-AP, önkalibráló flat)")
    p.add_argument("input", help="videófájl vagy képeket tartalmazó mappa")
    p.add_argument("-o", "--out", default="eredmeny", help="kimeneti mappa")
    p.add_argument("--percent", type=float, default=20, help="AP-nként a legjobb %% (alap: 20)")
    p.add_argument("--min-frames", type=int, default=5)
    p.add_argument("--candidates", type=float, default=50, help="videónál a globálisan legjobb %% megy tovább")
    p.add_argument("--half-box", type=int, default=32, help="AP-box félszélesség px-ben")
    p.add_argument("--step", type=int, default=0, help="AP-rácsosztás (0 = 1.5 * half-box)")
    p.add_argument("--search", type=int, default=16, help="lokális keresési sugár px-ben")
    p.add_argument("--no-refine", action="store_true", help="második (finomító) mérés kihagyása")
    p.add_argument("--no-flat", action="store_true", help="vinyettálás-korrekció kikapcsolása")
    p.add_argument("--flat-deg", type=int, default=3)
    p.add_argument("--no-gain", action="store_true")
    p.add_argument("--sharpen", type=str, default="2.0,1.5,1.2,1.0", help="wavelet rétegerősségek")
    p.add_argument("--no-distortion", action="store_true", help="torzítás-önkalibráció kikapcsolása")
    p.add_argument("--dist-deg", type=int, default=3, help="torzításpolinom foka (2-4)")
    p.add_argument("--stretch", action="store_true",
                   help="automatikus széthúzás fej-térrel (alapból ki: a fényesség = a bemenetével)")
    p.add_argument("--knee", type=float, default=0.85, help="lágy csúcsfény-levágás kezdete (0-1)")
    p.add_argument("--no-ca", action="store_true", help="kromatikus aberráció korrekciójának kikapcsolása")
    p.add_argument("--defringe", type=float, default=0.5, help="színszegély-csillapítás erőssége (0-1, 0 = ki)")
    p.add_argument("--mineral", type=float, default=0.0,
                   help="'mineral moon' kép készítése ezzel a színerősítéssel (pl. 4; 0 = nem készül)")
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    cfg = Config(stack_percent=a.percent, min_frames=a.min_frames, candidate_percent=a.candidates,
                 half_box=a.half_box, step=a.step, search=a.search, refine=not a.no_refine,
                 flat=not a.no_flat, flat_deg=a.flat_deg, gain=not a.no_gain,
                 distortion=not a.no_distortion, dist_deg=a.dist_deg, stretch=a.stretch, knee=a.knee,
                 ca=not a.no_ca, defringe=a.defringe, mineral=a.mineral,
                 sharpen=tuple(float(x) for x in a.sharpen.split(",")), workers=a.workers)
    run(a.input, a.out, cfg)


if __name__ == "__main__":
    main()
