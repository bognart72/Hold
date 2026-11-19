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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(os.path.expanduser("~"), ".holdstack_gui.json")
VIDEO_TYPES = [("Videó", "*.mp4 *.mov *.avi *.mkv *.ser *.m4v"), ("Minden fájl", "*.*")]
PROG_RE = re.compile(r"PROGRESS (\d+)/(\d+)")
TIME_RE = re.compile(r"^\[\s*([\d.]+)s\]\s*(.*)$")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("holdstack – Hold-stackelő")
        self.geometry("1280x860")
        self.minsize(980, 640)
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
        num(7, "Ásványi hold erőssége", "mineral", 4.0, 1.0, 10.0, 0.5, "ha be van kapcsolva")
        ttk.Label(opt, text="Élesítés (wavelet rétegek)").grid(row=8, column=0, sticky="w", pady=2)
        self.vars["sharpen"] = tk.StringVar(value="2.0,1.5,1.2,1.0")
        ttk.Entry(opt, textvariable=self.vars["sharpen"], width=16).grid(row=8, column=1, columnspan=2, sticky="w")

        self.flags = {}
        for r, (key, text, default) in enumerate([
                ("refine", "Finomító második mérés", True),
                ("flat", "Vinyettálás-korrekció (önkalibráló flat)", True),
                ("distortion", "Torzítás-önkalibráció (ferde telefon)", True),
                ("ca", "Kromatikus aberráció korrekció", True),
                ("gain", "Képkockánkénti fényesség-kiegyenlítés", True),
                ("mineral", "Ásványi hold (mineral moon) kép is készüljön", False),
                ("stretch", "Automatikus széthúzás (alapból ki – beéghet)", False)], start=9):
            v = tk.BooleanVar(value=default)
            ttk.Checkbutton(opt, text=text, variable=v).grid(row=r, column=0, columnspan=3, sticky="w")
            self.flags[key] = v

        # vezérlés
        ctl = ttk.Frame(left)
        ctl.pack(fill=tk.X)
        self.btn_run = ttk.Button(ctl, text="▶  Indítás", command=self._start)
        self.btn_run.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(ctl, text="■  Leállítás", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        ttk.Button(ctl, text="Kimeneti mappa megnyitása", command=self._open_out).pack(side=tk.LEFT)

        self.stage = tk.StringVar(value="Várakozás")
        ttk.Label(left, textvariable=self.stage, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.pbar = ttk.Progressbar(left, mode="determinate", maximum=100)
        self.pbar.pack(fill=tk.X)
        self.elapsed = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.elapsed, foreground="#666").pack(anchor="w")

        self.summary = tk.Text(left, height=12, width=56, wrap="word", font=("Consolas", 9),
                               relief="flat", background=self.cget("background"))
        self.summary.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

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
        self.img_tabs = {}
        for key, title in [("preview", "Eredmény"), ("mineral", "Ásványi hold"), ("aps", "Alignment pontok"),
                           ("torzitas", "Torzítási mező"), ("flat", "Vinyettálás")]:
            lbl = ttk.Label(self.nb, anchor="center")
            lbl.bind("<Configure>", lambda e, k=key: self._refit(k))
            self.nb.add(lbl, text=title)
            self.img_tabs[key] = [lbl, None]

    # ---------------------------------------------------------------- beállítások mentése
    def _load_settings(self):
        try:
            with open(SETTINGS, encoding="utf-8") as fh:
                s = json.load(fh)
        except Exception:
            return
        self.kind.set(s.get("kind", "video"))
        self.inp.set(s.get("input", ""))
        self.out.set(s.get("output", ""))
        for k, v in s.get("vars", {}).items():
            if k in self.vars:
                self.vars[k].set(v)
        for k, v in s.get("flags", {}).items():
            if k in self.flags:
                self.flags[k].set(v)

    def _save_settings(self):
        s = dict(kind=self.kind.get(), input=self.inp.get(), output=self.out.get(),
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
        if f["mineral"]: a += ["--mineral", v["mineral"]]
        if not f["refine"]: a.append("--no-refine")
        if not f["flat"]: a.append("--no-flat")
        if not f["distortion"]: a.append("--no-distortion")
        if not f["gain"]: a.append("--no-gain")
        if f["stretch"]: a.append("--stretch")
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
            self.img_tabs[k][0].configure(image="", text="")
        self.nb.select(0)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        cmd = self._args()
        self._append("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.btn_run.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.stage.set("Indul…")
        self.pbar["value"] = 0

    def _reader(self, proc):
        for raw in iter(proc.stdout.readline, b""):
            self.q.put(("line", raw.decode("utf-8", errors="replace")))
        proc.wait()
        self.q.put(("done", proc.returncode))

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
                    self._append(val)
                    m = PROG_RE.search(val)
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
        self.btn_stop.configure(state=tk.DISABLED)
        self.proc = None
        if code != 0:
            self.stage.set(f"Hiba vagy leállítás (kód {code})")
            return
        self.stage.set("Kész")
        self.pbar["value"] = 100
        self._show_results()

    # ---------------------------------------------------------------- eredmények
    def _show_results(self):
        out = self.out.get()
        base = os.path.splitext(os.path.basename(os.path.normpath(self.inp.get())))[0]
        files = dict(preview=f"{base}_preview.png", mineral=f"{base}_mineral.png", aps=f"{base}_aps.jpg",
                     torzitas=f"{base}_torzitas.png", flat=f"{base}_flat.png")
        for k, fn in files.items():
            p = os.path.join(out, fn)
            lbl = self.img_tabs[k][0]
            if os.path.exists(p):
                self.img_tabs[k][1] = Image.open(p).convert("RGB")
                self._refit(k)
            else:
                lbl.configure(image="", text="(ehhez a futáshoz nincs ilyen kép)")
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

    def _refit(self, key):
        lbl, img = self.img_tabs[key]
        if img is None:
            return
        w, h = max(lbl.winfo_width(), 50), max(lbl.winfo_height(), 50)
        s = min(w / img.width, h / img.height)
        im = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        ph = ImageTk.PhotoImage(im)
        self.photos[key] = ph
        lbl.configure(image=ph, text="")

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
