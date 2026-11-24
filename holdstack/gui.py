"""holdstack grafikus felület (tkinter).

Indítás:  python -m holdstack.gui   vagy dupla kattintás a holdstack_gui.pyw fájlon.

A feldolgozás külön folyamatban fut (python -m holdstack ...), a kimenete élőben látszik a
naplóablakban, a haladásjelző a "PROGRESS a/b" sorokból frissül. Leállítható.
A végén az eredményképek (élesített stack, AP-térkép, torzítási mező, flat) fülekre kerülnek.
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from .autotune import DEFAULTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(os.path.expanduser("~"), ".holdstack_gui.json")
VIDEO_TYPES = [("Videó", "*.mp4 *.mov *.avi *.mkv *.ser *.m4v"), ("Minden fájl", "*.*")]
PROG_RE = re.compile(r"PROGRESS (\d+)/(\d+)")
TIME_RE = re.compile(r"^\[\s*([\d.]+)s\]\s*(.*)$")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("holdstack – Hold- és Nap-stackelő")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(1280, sw - 40), min(860, sh - 90)
        self.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")
        self.minsize(min(980, w), min(600, h))
        self.proc = None
        self.q = queue.Queue()
        self.photos = {}
        self._build()
        self._load_settings()
        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- felület
    def _build(self):
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, padding=4)
        right = ttk.Frame(main, padding=4)
        main.add(left, weight=0)
        main.add(right, weight=1)

        # bemenet
        box = ttk.LabelFrame(left, text="Bemenet", padding=8)
        box.pack(fill=tk.X)
        self.kind = tk.StringVar(value="video")
        ttk.Radiobutton(box, text="Videófájl", variable=self.kind, value="video").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box, text="Képmappa (különálló fotók)", variable=self.kind, value="images").grid(
            row=0, column=1, columnspan=2, sticky="w")
        self.target = tk.StringVar(value="moon")
        tgt = ttk.Frame(box)
        tgt.grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(tgt, text="Célpont:").pack(side=tk.LEFT)
        ttk.Radiobutton(tgt, text="Hold", variable=self.target, value="moon").pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(tgt, text="Nap (fehér fényű napszűrő)", variable=self.target, value="sun").pack(side=tk.LEFT)
        self.inp = tk.StringVar()
        ttk.Entry(box, textvariable=self.inp, width=48).grid(row=1, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Button(box, text="Tallózás…", command=self._browse_in).grid(row=1, column=2, padx=(4, 0))
        ttk.Label(box, text="Kimeneti mappa:").grid(row=2, column=0, sticky="w")
        self.out = tk.StringVar()
        ttk.Entry(box, textvariable=self.out, width=48).grid(row=3, column=0, columnspan=2, sticky="we", pady=4)
        ttk.Button(box, text="Tallózás…", command=self._browse_out).grid(row=3, column=2, padx=(4, 0))
        box.columnconfigure(0, weight=1)

        # beállítások
        opt = ttk.LabelFrame(left, text="Beállítások", padding=8)
        opt.pack(fill=tk.X, pady=8)
        self.vars = {}

        def num(r, label, key, default, frm=0, to=100, inc=1, tip=""):
            ttk.Label(opt, text=label).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=str(default))
            ttk.Spinbox(opt, textvariable=v, from_=frm, to=to, increment=inc, width=8).grid(row=r, column=1, sticky="w")
            ttk.Label(opt, text=tip, foreground="#666").grid(row=r, column=2, sticky="w", padx=6)
            self.vars[key] = v

        num(0, "Legjobb képkockák (%)", "percent", 20, 1, 100, 1, "AP-nként")
        num(1, "Legalább ennyi kocka", "min_frames", 5, 1, 500, 1)
        num(2, "Jelöltek videónál (%)", "candidates", 50, 5, 100, 5, "≤60 kockánál mind")
        num(3, "AP-box félszélesség (px)", "half_box", 32, 12, 128, 4, "nagyobb = stabilabb")
        num(4, "Keresési sugár (px)", "search", 16, 4, 64, 2)
        num(5, "Csúcsfény-levágás (knee)", "knee", 0.85, 0.5, 1.0, 0.05, "beégés ellen")
        num(6, "Színszegély-csillapítás", "defringe", 0.5, 0.0, 1.0, 0.1, "0 = ki")
        num(7, "Ásványi hold erőssége", "mineral", 4.0, 1.0, 10.0, 0.5, "mindig elkészül")
        ttk.Label(opt, text="Wavelet rétegek (ha nincs dekonv.)").grid(row=8, column=0, sticky="w", pady=2)
        self.vars["sharpen"] = tk.StringVar(value="2.0,1.5,1.2,1.0")
        ttk.Entry(opt, textvariable=self.vars["sharpen"], width=16).grid(row=8, column=1, columnspan=2, sticky="w")

        num(9, "Nap: fáklyamező-kiemelés", "facula", 2.5, 0.0, 6.0, 0.5, "0 = ki; 5 fölött túl erős")
        num(10, "Dekonvolúció erőssége", "deconv_strength", 0.7, 0.0, 1.0, 0.1, "0,7 = mért optimum")
        self.flags = {}
        for r, (key, text, default) in enumerate([
                ("refine", "Finomító második mérés", True),
                ("flat", "Vinyettálás-korrekció (önkalibráló flat)", True),
                ("distortion", "Torzítás-önkalibráció (ferde telefon)", True),
                ("ca", "Kromatikus aberráció korrekció", True),
                ("deconv", "Dekonvolúciós élesítés (PSF a Hold pereméből)", True),
                ("gain", "Képkockánkénti fényesség-kiegyenlítés", True),
                ("trial", "Próbafuttatásos kalibráció a futás elején", False),
                ("stretch", "Automatikus széthúzás (alapból ki – beéghet)", False)], start=11):
            v = tk.BooleanVar(value=default)
            ttk.Checkbutton(opt, text=text, variable=v).grid(row=r, column=0, columnspan=3, sticky="w")
            self.flags[key] = v
        tune = ttk.Frame(opt)
        tune.grid(row=r + 1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.btn_tune = ttk.Button(tune, text="Gyors hangolás", command=self._start_tune)
        self.btn_tune.pack(side=tk.LEFT)
        self.btn_trial = ttk.Button(tune, text="Kalibráció próbafuttatással", command=self._start_trial)
        self.btn_trial.pack(side=tk.LEFT, padx=6)
        ttk.Button(tune, text="Alapértékek", command=self._reset_defaults).pack(side=tk.LEFT)

        # vezérlés
        ctl = ttk.Frame(left)
        ctl.pack(fill=tk.X)
        self.btn_run = ttk.Button(ctl, text="▶  Indítás", command=self._start)
        self.btn_run.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(ctl, text="■  Leállítás", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        ttk.Button(ctl, text="Eredmény betöltése", command=self._load_existing).pack(side=tk.LEFT)
        ttk.Button(ctl, text="Mappa", command=self._open_out).pack(side=tk.LEFT, padx=(6, 0))

        self.stage = tk.StringVar(value="Várakozás")
        ttk.Label(left, textvariable=self.stage, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.pbar = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.pbar.pack(fill=tk.X)
        self.elapsed = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.elapsed, foreground="#666").pack(anchor="w")

        sumf = ttk.Frame(left)
        sumf.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.summary = tk.Text(sumf, height=8, width=56, wrap="word", font=("Consolas", 9),
                               relief="flat", background=self.cget("background"))
        ssb = ttk.Scrollbar(sumf, command=self.summary.yview)
        self.summary.configure(yscrollcommand=ssb.set)
        ssb.pack(side=tk.RIGHT, fill=tk.Y)
        self.summary.pack(fill=tk.BOTH, expand=True)

        # jobb oldal: fülek
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)
        logf = ttk.Frame(self.nb)
        self.log = tk.Text(logf, wrap="none", font=("Consolas", 9), background="#111", foreground="#ddd",
                           insertbackground="#ddd")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(fill=tk.BOTH, expand=True)
        self.nb.add(logf, text="Napló")
        # A képek vásznon (Canvas) jelennek meg: a vászon nem veszi fel a kép méretét, így az
        # átméretezés nem vált ki újabb <Configure> eseményt (nincs végtelen méretezési ciklus).
        self.img_tabs = {}
        for key, title in [("preview", "Élesített"), ("wavelet", "Wavelet"), ("stack", "Stack"),
                           ("mineral", "Ásványi hold"), ("sun_enh", "Nap kiemelt"),
                           ("sun_det", "Nap részlet"),
                           ("aps", "Alignment pontok"), ("torzitas", "Torzítási mező"), ("flat", "Vinyettálás"),
                           ("dust", "Pormaszk")]:
            c = tk.Canvas(self.nb, background="#1e1e1e", highlightthickness=0, width=200, height=200)
            c.bind("<Configure>", lambda e, k=key: self._schedule_refit(k))
            self.nb.add(c, text=title)
            # [canvas, PIL kép, utolsó méret, időzítő-azonosító]
            self.img_tabs[key] = [c, None, None, None]

    # ---------------------------------------------------------------- beállítások mentése
    def _load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as fh:
                s = json.load(fh)
        except Exception:
            return
        self.kind.set(s.get("kind", "video"))
        self.target.set(s.get("target", "moon"))
        self.inp.set(s.get("input", ""))
        self.out.set(s.get("output", ""))
        for k, v in s.get("vars", {}).items():
            if k in self.vars:
                self.vars[k].set(v)
        for k, v in s.get("flags", {}).items():
            if k in self.flags:
                self.flags[k].set(v)

    def _save_settings(self):
        s = dict(kind=self.kind.get(), target=self.target.get(), input=self.inp.get(), output=self.out.get(),
                 vars={k: v.get() for k, v in self.vars.items()},
                 flags={k: v.get() for k, v in self.flags.items()})
        try:
            with open(SETTINGS, "w", encoding="utf-8") as fh:
                json.dump(s, fh, ensure_ascii=False, indent=1)
        except Exception:
            pass

    # ---------------------------------------------------------------- tallózás
    def _browse_in(self):
        if self.kind.get() == "video":
            p = filedialog.askopenfilename(title="Hold-videó kiválasztása", filetypes=VIDEO_TYPES)
        else:
            p = filedialog.askdirectory(title="Képeket tartalmazó mappa")
        if p:
            self.inp.set(os.path.normpath(p))
            if not self.out.get():
                name = os.path.splitext(os.path.basename(os.path.normpath(p)))[0]
                self.out.set(os.path.join(os.path.dirname(os.path.normpath(p)), "eredmeny", name))

    def _browse_out(self):
        p = filedialog.askdirectory(title="Kimeneti mappa")
        if p:
            self.out.set(os.path.normpath(p))

    # ---------------------------------------------------------------- alapértékek, hangolás
    def _reset_defaults(self):
        for k in ("percent", "min_frames", "candidates", "half_box", "search", "knee", "defringe", "sharpen"):
            self.vars[k].set(str(DEFAULTS[k]))
        self.vars["mineral"].set("4.0")
        self.vars["facula"].set("2.5")
        for k in ("refine", "flat", "distortion", "ca", "gain"):
            self.flags[k].set(bool(DEFAULTS[k]))
        self.vars["deconv_strength"].set("0.7")
        self.flags["deconv"].set(True)
        self.flags["stretch"].set(False)
        self.flags["trial"].set(False)
        self._save_settings()
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", "Az alapértékek visszaállítva.")
        self.stage.set("Alapértékek")
        self.pbar["value"] = 0
        self.elapsed.set("")

    def _start_tune(self):
        inp = self.inp.get()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("holdstack", "Előbb válassz videót vagy képmappát.")
            return
        self._tune_json = os.path.join(os.path.expanduser("~"), ".holdstack_tune.json")
        try:
            os.remove(self._tune_json)
        except OSError:
            pass
        self._launch([sys.executable, "-u", "-m", "holdstack.autotune", inp, "--json", self._tune_json], job="tune")
        self.stage.set("Hangolás: a felvétel elemzése…")

    def _start_trial(self):
        inp, out = self.inp.get(), self.out.get()
        if not inp or not os.path.exists(inp) or not out:
            messagebox.showerror("holdstack", "Előbb válassz bemenetet és kimeneti mappát.")
            return
        base = os.path.splitext(os.path.basename(os.path.normpath(inp)))[0]
        self._trial_json = os.path.join(out, f"{base}_kalibracio.json")
        try:
            os.remove(self._trial_json)
        except OSError:
            pass
        cmd = [a for a in self._args() if a != "--trial"] + ["--trial-only"]
        self._launch(cmd, job="trial")
        self.stage.set("Kalibráció próbafuttatással…")

    def _apply_trial(self):
        try:
            with open(self._trial_json, encoding="utf-8") as fh:
                t = json.load(fh)
        except Exception as e:
            messagebox.showerror("holdstack", f"A kalibráció eredménye nem olvasható: {e}")
            return
        b = t["best"]
        self.vars["half_box"].set(str(b["half_box"]))
        self.vars["percent"].set(f"{b['percent']:g}")
        self._save_settings()
        rows = sorted(t["trials"], key=lambda r: -r["score"])
        lines = [f"Kalibráció kész ({t['frames']} kocka, csempe {t['tile'][2]}x{t['tile'][3]} px).",
                 f"Legjobb: AP-box {2 * b['half_box']} px (félszélesség {b['half_box']}), "
                 f"legjobb {b['percent']:g}%  (+{t['gain_percent']:.1f}% a kiinduló beállításhoz képest)", "",
                 "box = teljes AP-box (px), a mezőben a félszélesség áll", "",
                 " box   arány   pontszám  relatív   részlet     zaj"]
        for r in rows:
            lines.append(f"{2 * r['half_box']:4d}  {r['percent']:5.0f}%  {r['score']:9.4f}  {r['relative'] * 100:6.1f}%"
                         f"  {r['detail']:.3e}  {r['noise']:.4f}")
        lines.append("\nA mezőkbe beírtam a legjobb beállítást; az Indítással ezzel fut a teljes feldolgozás.")
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", "\n".join(lines))
        self.stage.set("Kalibráció kész – a beállítások a mezőkben")

    def _apply_tune(self):
        try:
            with open(self._tune_json, encoding="utf-8") as fh:
                t = json.load(fh)
        except Exception as e:
            messagebox.showerror("holdstack", f"A hangolás eredménye nem olvasható: {e}")
            return
        p = t["params"]
        for k in ("percent", "min_frames", "candidates", "half_box", "search", "knee", "defringe", "sharpen"):
            self.vars[k].set(str(p[k]))
        for k in ("flat", "distortion"):
            self.flags[k].set(bool(p[k]))
        self._save_settings()
        self.summary.delete("1.0", tk.END)
        self.summary.insert("1.0", "A felvételhez hangolt beállítások betöltve:\n\n- " + "\n- ".join(t["reasons"]))
        self.stage.set("Hangolás kész – a beállítások a mezőkben")

    def _load_existing(self):
        """Egy korábbi futás eredményének megjelenítése a kimeneti mappából (újrafuttatás nélkül)."""
        base = os.path.splitext(os.path.basename(os.path.normpath(self.inp.get())))[0]
        rp = os.path.join(self.out.get(), f"{base}_report.json")
        if not os.path.exists(rp):
            messagebox.showinfo("holdstack", f"Nincs eredmény ebben a mappában:\n{rp}")
            return
        self.summary.delete("1.0", tk.END)
        self.stage.set("Korábbi eredmény betöltve")
        self._show_results()

    def _open_out(self):
        p = self.out.get()
        if p and os.path.isdir(p):
            os.startfile(p)

    # ---------------------------------------------------------------- futtatás
    def _args(self):
        v = {k: x.get().strip() for k, x in self.vars.items()}
        a = [sys.executable, "-u", "-m", "holdstack", self.inp.get(), "-o", self.out.get(),
             "--percent", v["percent"], "--min-frames", v["min_frames"], "--candidates", v["candidates"],
             "--half-box", v["half_box"], "--search", v["search"], "--knee", v["knee"], "--sharpen", v["sharpen"],
             "--defringe", v["defringe"]]
        f = {k: x.get() for k, x in self.flags.items()}
        if not f["ca"]: a.append("--no-ca")
        if not f.get("deconv", True): a += ["--sharpen-mode", "wavelet"]
        a += ["--mineral", v["mineral"], "--deconv-strength", v.get("deconv_strength") or "0.7"]
        if not f["refine"]: a.append("--no-refine")
        if not f["flat"]: a.append("--no-flat")
        if not f["distortion"]: a.append("--no-distortion")
        if not f["gain"]: a.append("--no-gain")
        if f["stretch"]: a.append("--stretch")
        if f.get("trial"): a.append("--trial")
        a += ["--target", self.target.get()]
        if self.target.get() == "sun":
            a += ["--facula", v.get("facula") or "2.5"]
        return a

    def _start(self):
        inp, out = self.inp.get(), self.out.get()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("holdstack", "Válassz létező videót vagy képmappát.")
            return
        if self.kind.get() == "images" and not os.path.isdir(inp):
            messagebox.showerror("holdstack", "Képmappa módban mappát kell választani.")
            return
        if not out:
            messagebox.showerror("holdstack", "Adj meg kimeneti mappát.")
            return
        self._save_settings()
        self.log.delete("1.0", tk.END)
        self.summary.delete("1.0", tk.END)
        for k in self.img_tabs:
            self.img_tabs[k][1] = None
            self.img_tabs[k][2] = None
            self.img_tabs[k][0].delete("all")
        self._launch(self._args(), job="stack")
        self.stage.set("Indul…")

    def _launch(self, cmd, job):
        """Külön folyamat indítása (stack vagy hangolás), élő naplóval."""
        self.job = job
        self.log.delete("1.0", tk.END)
        self.nb.select(0)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self._append("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        # A napló FÁJLBA megy, a felület onnan olvassa: csővel (pipe) a feldolgozás a kiírásnál
        # várakozhat, ha a felület épp nem üríti a kis Windows-csőpuffert (mérve: ~25% lassulás).
        # A folyamat leválasztva indul (DETACHED_PROCESS), ne örökölje a GUI ütemezési állapotát.
        import tempfile
        self._logpath = os.path.join(tempfile.gettempdir(), f"holdstack_gui_{os.getpid()}_{int(time.time())}.log")
        logf = open(self._logpath, "wb")
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, creationflags=flags)
        logf.close()  # a gyermek saját példánya nyitva marad
        threading.Thread(target=self._reader, args=(self.proc, self._logpath), daemon=True).start()
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_tune.configure(state=tk.DISABLED)
        self.btn_trial.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.pbar["value"] = 0

    def _reader(self, proc, path):
        """A naplófájl követése (tail) a folyamat végéig; teljes sorokat ad tovább."""
        buf = b""
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read()
                if chunk:
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for raw in lines:
                        self.q.put(("line", raw.rstrip(b"\r").decode("utf-8", errors="replace") + "\n"))
                elif proc.poll() is not None:
                    rest = fh.read()
                    buf += rest
                    if not rest:
                        break
                else:
                    time.sleep(0.2)
        if buf:
            self.q.put(("line", buf.decode("utf-8", errors="replace") + "\n"))
        self.q.put(("done", proc.returncode))
        try:
            os.remove(path)
        except OSError:
            pass

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.stage.set("Leállítva")

    def _append(self, s):
        self.log.insert(tk.END, s)
        self.log.see(tk.END)

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "line":
                    m = PROG_RE.search(val)
                    self._append(PROG_RE.sub("", val).rstrip() + "\n")
                    t = TIME_RE.match(val.strip())
                    if t:
                        self.elapsed.set(f"Eltelt idő: {float(t.group(1)):.0f} s")
                    if m:
                        a, b = int(m.group(1)), int(m.group(2))
                        self.pbar["value"] = 100.0 * a / max(b, 1)
                        name = val.split("PROGRESS")[0]
                        name = TIME_RE.match(name.strip()).group(2) if TIME_RE.match(name.strip()) else name
                        self.stage.set(name.split(":")[0].strip().capitalize() + f"  ({a}/{b})")
                    elif t:
                        msg = t.group(2)
                        self.stage.set(msg.split(":")[0].strip().capitalize()[:60])
                else:
                    self._finished(val)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _finished(self, code):
        self.btn_run.configure(state=tk.NORMAL)
        self.btn_tune.configure(state=tk.NORMAL)
        self.btn_trial.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.proc = None
        if code != 0:
            self.stage.set(f"Hiba vagy leállítás (kód {code})")
            return
        self.pbar["value"] = 100
        if getattr(self, "job", "stack") == "tune":
            self._apply_tune()
            return
        if self.job == "trial":
            self._apply_trial()
            return
        self.stage.set("Kész")
        self._show_results()

    # ---------------------------------------------------------------- eredmények
    def _show_results(self):
        out = self.out.get()
        base = os.path.splitext(os.path.basename(os.path.normpath(self.inp.get())))[0]
        files = dict(preview=f"{base}_elesitett.png", wavelet=f"{base}_wavelet.png", stack=f"{base}_stack.png",
                     mineral=f"{base}_mineral.png",
                     sun_enh=f"{base}_nap_kiemelt.png", sun_det=f"{base}_nap_reszlet.png",dust=f"{base}_pormaszk.png", aps=f"{base}_aps.jpg",
                     torzitas=f"{base}_torzitas.png", flat=f"{base}_flat.png")
        for k, fn in files.items():
            p = os.path.join(out, fn)
            if os.path.exists(p):
                im = Image.open(p).convert("RGB")
                im.thumbnail((2000, 2000))  # a megjelenítéshez elég; gyorsabb újraméretezés
                self.img_tabs[k][1] = im
                self.img_tabs[k][2] = None
                self._refit(k, force=True)
            else:
                self.img_tabs[k][1] = None
                self._set_message(k, "(ehhez a futáshoz nincs ilyen kép)")
        self.nb.select(1)
        rp = os.path.join(out, f"{base}_report.json")
        try:
            with open(rp, encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:
            return
        d, b, fl = r.get("distortion", {}), r.get("brightness", {}), r.get("flat", {})
        lines = [
            f"Bemenet: {r['kind']}, {r['frames']} kocka, {r['size'][0]}x{r['size'][1]}",
            f"Referencia: {r['reference']}",
            f"Jelölt / felhasznált kocka: {r['candidates']} / {r['contributing']}",
            f"AP-k: {r['aps']} (lépés {r['ap_step']} px), sikeres mérés {r['ap_success'] * 100:.1f}%",
            f"Lokális eltolás: medián {r['local_shift_px']['median']:.2f} px, p95 {r['local_shift_px']['p95']:.2f} px",
        ]
        if fl.get("enabled"):
            lines.append(f"Vinyettálás: {fl['g_min']:.3f} … {fl['g_max']:.3f}")
        if d.get("enabled"):
            lines.append(f"Torzítás: maradék {d['rms_before']:.2f} → {d['rms_after']:.2f} px rms, "
                         f"max korrekció a Holdon {d.get('max_shift_moon_px', 0):.1f} px")
        else:
            lines.append(f"Torzítás: nem alkalmazva ({d.get('msg', '')})")
        tc = r.get("trial_calibration")
        if tc:
            tb = tc["best"]
            # a kalibrált értékek a mezőkbe (egy következő, kalibráció nélküli futás is ezt használja)
            self.vars["half_box"].set(str(tb["half_box"]))
            self.vars["percent"].set(f"{tb['percent']:g}")
            self._save_settings()
            lines.append(f"Kalibráció: AP-box {2 * tb['half_box']} px (félszélesség {tb['half_box']}), "
                         f"legjobb {tb['percent']:g}% – pontszám +{tc['gain_percent']:.1f}% a kiindulóhoz képest "
                         f"({len(tc['trials'])} próba)")
        du = r.get("dust") or {}
        if r.get("target") == "sun" or du.get("enabled"):
            lines.append(f"Célpont: {'Nap' if r.get('target') == 'sun' else 'Hold'}; pormaszk: "
                         + (f"{du['specks']} porszem eltávolítva (legmélyebb {du['max_depth'] * 100:.0f}%)"
                            if du.get("enabled") else f"nincs ({du.get('msg', '')})"))
        sp = r.get("sharpening") or {}
        if sp.get("mode") == "auto":
            lines.append(f"Élesítés: dekonvolúció, PSF szigma a Hold pereméből {sp['psf_sigma']:.2f} px")
        elif sp:
            lines.append("Élesítés: wavelet " + ",".join(f"{x:g}" for x in sp.get("amounts", []))
                         + (f" ({sp['msg']})" if sp.get("msg") else ""))
        sd = r.get("sun_detail") or {}
        if sd.get("enabled"):
            ps = sd.get("psf_sigma")
            lines.append("Nap részlet: " + (f"PSF szigma {ps:.2f} px, foltok dekonvolválva" if ps else
                                            "PSF nem mérhető, dekonvolúció nélkül")
                         + f"; fáklyamező-kiemelés {sd.get('facula_gain', 0):g}")
        ca = r.get("chromatic_aberration", {}).get("per_frame", {})
        if ca.get("enabled"):
            lines.append(f"Kromatikus aberráció: R {ca['R']['mean_shift_px']:.2f} px (max {ca['R']['max_shift_px']:.2f}), "
                         f"B {ca['B']['mean_shift_px']:.2f} px (max {ca['B']['max_shift_px']:.2f}) – korrigálva")
        if b:
            lines.append(f"Beégett pixel: bemenet max {b['input_saturated_max'] * 100:.3f}%, "
                         f"stack {b['stack_saturated'] * 100:.3f}%, élesített {b['sharpened_saturated'] * 100:.3f}%")
        lines.append(f"Futásidő: {r['seconds']:.0f} s")
        lines.append(f"\nFájlok: {out}")
        self.summary.insert("1.0", "\n".join(lines))

    def _schedule_refit(self, key):
        """Késleltetett újraméretezés: ablakhúzás közben csak egyszer, a végén fut le."""
        t = self.img_tabs[key]
        if t[3] is not None:
            self.after_cancel(t[3])
        t[3] = self.after(150, lambda: self._refit(key))

    def _set_message(self, key, text):
        c = self.img_tabs[key][0]
        c.delete("all")
        c.create_text(c.winfo_width() // 2, c.winfo_height() // 2, text=text, fill="#bbb")

    def _refit(self, key, force=False):
        t = self.img_tabs[key]
        c, img = t[0], t[1]
        t[3] = None
        if img is None:
            return
        w, h = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        if not force and t[2] == (w, h):
            return  # nincs valódi méretváltozás
        t[2] = (w, h)
        s = min(w / img.width, h / img.height)
        im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
        ph = ImageTk.PhotoImage(im)
        self.photos[key] = ph
        c.delete("all")
        c.create_image(w // 2, h // 2, image=ph, anchor="center")

    def _on_close(self):
        if self.proc and self.proc.poll() is None:
            if not messagebox.askyesno("holdstack", "Fut a feldolgozás. Leállítod és kilépsz?"):
                return
            self.proc.terminate()
        self._save_settings()
        self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
