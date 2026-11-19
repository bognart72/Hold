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

## Használat

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
| `--sharpen` | 2.0,1.5,1.2,1.0 | wavelet-rétegek erőssége (1 = változatlan) |

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
5. **AP-rács:** lépcsőzetes, fényesség-, lefedettség- és struktúra-szűréssel.
6. **Lokális mérés:**
   - Zero-mean NCC (`matchTemplate`), szubpixeles parabola.
   - Kiugrók szűrése a szomszédok mediánjával.
   - AP-nkénti élesség az **eredeti**, nem interpolált képkockán.
7. **Finomítás:** AP-alapú élesebb referencia, majd újramérés.
8. **Stack:**
   - AP-nként a legjobb N% lágy kiválasztással.
   - A homográfia és a sima lokális eltolási mező **egyetlen** interpolációban (bicubic remap).
   - Pixelenkénti súlyozott átlag érvényességi maszkkal. Nincsenek patch-varratok.
9. **Mentés:** kivágás, 16 bites TIFF, wavelet-élesítés.

Részletes háttér, szakirodalom és mérések: `KUTATAS_es_TERV.md`.
