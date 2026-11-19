# Hold

Hold fotó/videó feldolgozás (Dobson távcső + mobiltelefon)

## holdstack – Hold-stackelő telefonos okulárfelvételekhez

Mobiltelefonnal, afokálisan (okulárra tartva), **nem követő** Dobsonnal készült Hold-videóból
vagy fotósorozatból egyetlen élesebb, zajmentesebb képet készít az AutoStakkert!3 / PlanetarySystemStacker
mintájára (lucky imaging, alignment pointok).

Kifejezetten a telefonos felvétel hibáira készült:

- Drift: a Hold vándorol a képen, részben ki is lóghat belőle.
- Ferde, nem tengelyes telefon: látómező-függő torzítás és vinyettálás. Ezek önkalibrációval, magából a felvételből becsülhetők.
- Látszó okulár-blende: kimaszkolja.
- Rezgés és seeing miatti helyi hullámzás.
- Nagy ugrások a fotók között.

## Telepítés

```powershell
pip install -r requirements.txt
```

## Grafikus felület

Dupla kattintás a `holdstack_gui.pyw` fájlon, vagy:

```powershell
python -m holdstack.gui
```

A felület részei:
- Bemenet: videófájl vagy képmappa, és a kimeneti mappa tallózással.
- Az összes fontos beállítás. A felület a legutóbbi értékeket megjegyzi (`~/.holdstack_gui.json`).
- **Indítás / Leállítás**, élő napló, szakaszonkénti haladásjelző és eltelt idő.
- A végén összefoglaló és fülek az eredményképekkel: élesített stack, alignment pontok, torzítási mező, vinyettálás.

A feldolgozás külön folyamatban fut, ugyanazzal a parancssori programmal.

## Használat parancssorból

```powershell
python -m holdstack Video\h001.mp4 -o eredmeny\h001
```

```powershell
python -m holdstack Teszt_kepek -o eredmeny\Teszt_kepek
```

A bemenet lehet videófájl (mp4/mov/avi…) vagy képeket tartalmazó mappa (jpg/png/tif).
A képeket az EXIF-időbélyeg szerint rendezi, és alkalmazza az EXIF-forgatást.

### Fontosabb opciók

| Opció | Alap | Jelentés |
|---|---|---|
| `--percent` | 20 | AP-nként a legjobb ennyi % kerül a stackbe |
| `--min-frames` | 5 | de legalább ennyi képkocka |
| `--candidates` | 50 | videónál a globálisan legjobb % megy a részletes feldolgozásba (≤ 60 kockánál mind) |
| `--half-box` | 32 | AP-box félszélessége px-ben (nagyobb = stabilabb, kisebb = finomabb torzításkövetés) |
| `--search` | 16 | lokális keresési sugár px-ben |
| `--no-refine` | | a finomító második mérés kihagyása (gyorsabb) |
| `--no-flat` | | a vinyettálás-korrekció kikapcsolása |
| `--sharpen` | 2.0,1.5,1.2,1.0 | wavelet-rétegek erőssége (1 = változatlan), csak a fényességcsatornán |
| `--no-distortion` | | a szenzorhoz kötött torzítás önkalibrációjának kikapcsolása |
| `--dist-deg` | 3 | a torzításpolinom foka (2–4) |
| `--stretch` | | automatikus széthúzás fej-térrel; **alapból ki**, a fényesség megegyezik a bemenetével |
| `--knee` | 0.85 | lágy csúcsfény-levágás kezdete az élesített képen |
| `--no-ca` | | a kromatikus aberráció korrekciójának kikapcsolása |
| `--defringe` | 0.5 | színszegély-csillapítás erőssége az élesített képen (0–1) |
| `--mineral` | 0 | „mineral moon” (ásványi hold) kép készítése ezzel a színerősítéssel, pl. `--mineral 5` |

### Kromatikus aberráció

1. **Képkockánként, a stack előtt:**
   - Rácspontokon méri, mennyire csúszik el a piros és a kék csatorna a zöldhöz képest (zero-mean NCC, szubpixel). Legfeljebb 16 képkockából, időben egyenletesen.
   - Ebből a szenzorhoz kötött, robusztus 2. fokú polinom-mezőt illeszt, és **minden képkockát** ezzel korrigál.
   - Így a vándorló Hold különböző helyzeteiben más-más színhiba nem kenődik össze.
2. **A kész stacken:** egy második, lineáris mező a maradékot és a légköri diszperziót veszi ki.
3. **Szegélycsillapítás** (`--defringe`): az erős fényesség-éleknél a krominanciát a környezeti átlag felé húzza. A fényesség érintetlen marad.

Megjegyzés: a Samsung telefon JPEG-jei már a telefonban CA-korrigáltak (a mért maradék ≤ 0,15 px). A videókockáknál és a nyers felvételeknél nagyobb lehet.

### Ásványi hold (mineral moon)

Opcionális külön kép: `*_mineral.tif` és `*_mineral.png`.
- A fényesség az élesített képé.
- A színinformáció lépései:
  1. Zajmentesítés: normált konvolúció, csak a megvilágított felszínből.
  2. A nagy léptékű színátmenet levonása, 2. fokú polinommal. Ez a telefon színvinyettálása, a fehéregyensúly és a légkör hatása, nem ásványi szín.
  3. Erősítés a megadott szorzóval.
- A perem és a terminátor felé a szín fokozatosan halványul, mert ott a színhiba és a háttér-bekeverés miatt megbízhatatlan.
- A valódi regionális különbségek megmaradnak: a titánban gazdag mare-k kékesek, a vasban gazdagok narancsosak.

### Fényesség: miért nem ég be

- A stack **súlyozott átlag**, ezért a fényessége megegyezik a bemeneti képekével. Nincs automatikus széthúzás.
- A vinyettálás-flat úgy van normálva, hogy a Hold által bejárt területen az átlaga 1 legyen. A képkockánkénti erősítések átlaga is 1. A korrekció tehát összességében nem világosít.
- Az élesítés csak a fényességet erősíti (a színarány marad). A túllövéseket lágy csúcsfény-levágás (tanh roll-off) kezeli, nem kemény vágás.
- A futás a bemenet, a stack és az élesített kép beégett pixelarányát is kiírja és a jelentésbe menti.

### Kimenet (`-o` mappa)

- `*_stack.tif`: 16 bites, élesítetlen stack. Ezt érdemes továbbvinni (Registax, AstroSurface, PixInsight).
- `*_stack_sharpened.tif`, `*_preview.png`: alap wavelet-élesítéssel.
- `*_aps.jpg`: az alignment pointok a referencián. Zöld = megbízható, piros = gyakran sikertelen.
- `*_flat.png`: a becsült vinyettálás.
- `*_report.json`: minden paraméter, képkockánkénti élesség, illesztés, erősítés és súly.

## Algoritmus röviden

1. **Elemzés:**
   - Kicsinyített kép, Hold-maszk (az égi háttér és a blende kimarad).
   - Zajtűrő élesség: Gauss utáni Laplace-variancia / fényesség².
2. **Globális illesztés:**
   - AKAZE jellemzőpontok + RANSAC a durva becsléshez (nagy ugrás és forgás is mehet).
   - ECC **homográfia** két szintű piramison, csak a Hold-maszkon.
3. **Önkalibráló flat:**
   - A szenzorhoz rögzített vinyettálás 3. fokú polinomként, robusztus (Huber) illesztéssel.
   - Képkockánkénti erősítés-kiegyenlítés.
4. **Referencia:** a legjobb 5% átlaga.
5. **Torzítás-önkalibráció:**
   - A telefon ferdesége és az okulár miatt a kép a szenzorhoz kötötten, nemlineárisan torzul.
   - Modell: u = x + s·Σ c·monom((x−c)/s), 2–3. fokú tagok, x és y komponensre is.
   - A pontpárok a ref-pont és a képkockabeli helye (globális homográfia + mért AP-eltolás), legfeljebb 60 képkockából, időben egyenletesen.
   - Becslés „variable projection” módszerrel: adott c mellett képkockánként zárt alakú homográfia, a maradékra robusztus (Huber) legkisebb négyzetek.
   - Csak akkor alkalmazza, ha a maradékot ≥ 5%-kal csökkenti.
   - Utána a referencia és az AP-rács a torzítatlan térben épül újra, így a kimenet geometriája „kivasalt”.
6. **AP-rács:** lépcsőzetes, fényesség-, lefedettség- és struktúra-szűréssel.
7. **Lokális mérés:**
   - Zero-mean NCC (`matchTemplate`), szubpixeles parabola.
   - Kiugrók szűrése a szomszédok mediánjával.
   - AP-nkénti élesség az **eredeti**, nem interpolált képkockán.
8. **Finomítás:** AP-alapú élesebb referencia, majd újramérés.
9. **Stack:**
   - AP-nként a legjobb N% lágy kiválasztással.
   - A lokális eltolási mező, a homográfia és a torzítás **egyetlen** interpolációban (bicubic remap).
   - Pixelenkénti súlyozott átlag érvényességi maszkkal. Nincsenek patch-varratok.
10. **Mentés:** kivágás, 16 bites TIFF az eredeti fényességi skálán, wavelet-élesítés lágy csúcsfény-levágással.

Részletes háttér, szakirodalom és mérések: `KUTATAS_es_TERV.md`.

## Eredmények v0.2-vel

v0.2 újdonságai: torzítás-önkalibráció, fényességmegőrzés, kromatikus aberráció, ásványi hold, grafikus felület.

**Videó** (21 s, 620 kocka, 4K):
- A torzítás-korrekció a Holdon legfeljebb 24,6 px-ig egyenesít, a pontpár-maradék 0,78 → 0,49 px.
- A helyi maradék eltolás p95 értéke 2,7 → 2,0 px.
- Nincs beégett pixel. Az élesített kép 99,9%-os fényességszintje 0,82; v0.1-ben teljes fehérre volt húzva.
- A videót a CA-korrekció és az ásványi hold beépítése előtt futtattam, így ezek eredménye itt nincs.

**17 fotó:**
- A torzítás-korrekció legfeljebb 41 px a Holdon, a pontpár-maradék 1,43 → 0,62 px.
- A kromatikus aberráció ≤ 0,1 px, mert a telefon JPEG-je már korrigált. A program ezt is kijavítja.
- Nincs beégés. Ásványi hold kép készült `--mineral 5` erősséggel.

## Eddigi eredmények (v0.1)

| Bemenet | Felvétel | Eredmény | Futásidő |
|---|---|---|---|
| Videó, 21 s, 620 kocka, 4K H.264, Samsung | Dobson + telefon az okulárban | 310 jelölt, 881 AP, 100% sikeres AP-mérés, a lokális eltolás mediánja 0,77 px (p95: 2,7 px). Nincs dupla kontúr, részletgazdagabb, mint a legjobb egyedi képkocka. | ~23 perc |
| 17 JPG, 4000×3000, digitális zoom 1,72× | ugyanaz, egyedi fotók | 956 AP, 99,9% sikeres mérés, a becsült vinyettálás 0,75…1,14. Nincs dupla kontúr, kb. 20%-kal kisebb a zaj. | ~2,5 perc |

**Ismert korlátok és következő lépések:**
- Lassú 4K videón (~23 perc).
  - A globális ECC illesztés ritkítható: szomszédos képkockák kezdőértéke, kisebb piramis.
  - Az AP-mérés vektorizálható vagy GPU-ra vihető.
- A szenzorhoz kötött nemlineáris torzítást most a homográfia és a lokális AP-mező együtt kezeli. A külön torzítás-önkalibráció még hiányzik.
- Szintetikus tesztvideó-generátor a pixelpontosság számszerű ellenőrzéséhez.
- RAW/DNG bemenet, sigma-clipping, drizzle, RGB-csatornák lokális igazítása, grafikus felület.
