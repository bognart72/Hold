# Hold-stackelő: kutatás és algoritmus-terv

Felvétel: mobiltelefon afokálisan a Dobson okulárján, 10–60 s videó, a Dobson nem követ.
Cél: a legjobb ~20% kiválasztása, pixelpontos illesztés, stack. A minta az AutoStakkert!3 (AS!3).

---

## 1. Mit tudunk a mintaprogramokról

### AutoStakkert!3 (zárt forrású, Emil Kraaikamp)
Az AS!3 forráskódja nem nyilvános. A kézikönyvből és a fórumokból ismert működés:
- **Image stabilization:** *Planet* mód (fényesség-súlypont, COG) vagy *Surface* mód (egy jól strukturált ablak korrelációja). Holdra a *Surface* mód kell.
- **Quality estimator:** *Local* vagy *Global*, *Gradient* vagy *Edge* módszer. A **Noise Robust** paraméter egy Gauss-elmosás mértéke, ez véd az ellen, hogy a zajt élességnek mérje. A „Laplace delta” a minőségmérték neve.
- **Alignment pointok (AP):** rács, AP-méret 24–200 px. Holdon kb. 100 px-es AP-k jól működnek. Kisebb AP-val finomabb a deformáció követése, de instabilabb.
- **Lokális választás:** AP-nként külön a legjobb N%, a stack ebből áll össze.
- Opciók: drizzle 1.5x/3x, sigma-clip, normalizálás.

### PlanetarySystemStacker (PSS) – nyílt forrású, Python + OpenCV
A szerző tesztjeiben AS!3-mal egyenértékű eredményt ad, tehát **Pythonban is elérhető az AS!3 minősége**. Az algoritmus-leírás (v0.9.8) lépései:

1. **Előfeldolgozás:** mono kép, Gauss-elmosott kép (`frames_gauss_width`), annak Laplace-a. A shift-keresés az elmosott képen fut, így a pixelzaj nem okoz hamis minimumot.
2. **Globális rangsor:** `Var(Δl)`, azaz Laplace-variancia, opcionálisan 2-es stride-dal. A legjobb képkocka lesz a referencia.
3. **Globális igazítás, Surface mód:**
   - Ablakválasztás a merit = min(Σ|∂y l / l|, Σ|∂x l / l|) alapján, csak a küszöb feletti pixeleken. Így a választott ablakban x- és y-irányú struktúra is van.
   - **MultiLevelCorrelation:** NCC előbb 2-es stride-dal, a keresési sugár `(search_width-4)/2`, utána finom rácson ±4 px. Ha az optimum a keresési tartomány szélére esik, a mérés sikertelen.
   - **Driftkezelés:**
     - Az előző képkocka eltolása az induló becslés.
     - A referenciától visszafelé, majd előre halad.
     - Ha az ablak a kép széléhez ér, elmozdul.
     - Ha egy ablak elbukik, a következő legjobb ablakkal újrapróbálja.
4. **Átlagkép:** a legjobb X% globálisan eltolva, a közös metszeten (`intersection_shape`).
5. **AP-rács:**
   - Lépcsőzetes rács, box és nagyobb patch.
   - Feltételek: fényesség, kontraszt (max−min) és struktúra (min(sx, sy), normalizálva).
   - A limbus melletti AP a fényes pixelek súlypontja felé tolódik.
6. **AP-nkénti rangsor:** Laplace az AP-boxon, AP-nként saját best_frame lista.
7. **Lokális shift:** MultiLevelCorrelation **büntetőtaggal**: p = 1 − f·((sy/swf−1)² + (sx/swf−1)²). Ez a nagy, valószínűtlen eltolásokat bünteti.
8. **Stack:**
   - Teljes eltolás = globális + lokális. Ez merev patch-eltolás (`remap_rigid`) az eredeti színes képen.
   - **Sátor (tent) súlyok:** 0 a patch szélén, 1 a közepén. Az eredmény Σ(súly·patch)/Σsúly.
   - Ahol nincs AP, ott háttérkép áll be. A foreground_weight = min(1, ssfw/bt) maszkkal keveredik.
9. **Utófeldolgozás:**
   - RGB-csatornák igazítása (szubpixel, MultiLevelCorrelation).
   - Többszintű unsharp masking Gauss- és bilaterális szűrővel. Ha a bilaterális arány 0, ez a **Registax 6 wavelet**-algoritmusa.

### Siril multipoint (a PSS szerzőjének segítségével készült)
Ami a PSS-en felül van benne:
- **Referencia** a legjobb 5% átlagából, majd **finomító menet:** AP-nként a legjobb képkockákból új, élesebb referencia készül, és minden shiftet újramér.
- **Fényesség-normált minőség:** a Laplace-pontszám osztva az átlagfényességgel. Így az átlátszóság vagy a fényesség pulzálása nem tűnik élességnek.
- **Shift smoothing (2.5 rácsegység):** robusztus lokális illesztés a szomszédos AP-k shiftjeire, ez kiszűri a mérési zajt.
- **Zero-mean NCC** opció: erősítés- és offset-invariáns.
- **Lágy kiválasztás:** a vágási határ körül cosinus-átmenet, így a szomszédos AP-k hasonló képkockákat használnak.
- **Warp engine:** az AP-shiftekből sima, sűrű eltolási mező, és minden kép egyszer interpolálódik. Nincsenek patch-varratok.
- **Skip failed AP shifts:** a sikertelen mérés nem nulla eltolással kerül be, hanem kimarad. Ez kiküszöböli a szellemképet.
- Alapértékek: box félszélesség 24, rácsosztás 54, lokális keresés 14 px, globális keresés 34 px (legfeljebb 150).

### Szakirodalom (légköri turbulencia, több képkockás helyreállítás)
- **Lucky region fusion** (Aubailly és mtsai.): helyi élességmérték alapján kis „szerencsés” régiók összefésülése. Ez az AP-nkénti választás elméleti alapja.
- **Blokkalapú illesztés és lucky súlyozás** (Joshi & Cohen): blokkonkénti helyi igazítás, majd súlyozott fúzió.
- **Nemmerev regisztráció és dekonvolúció** (Milanfar-csoport, Zhu & Milanfar): előbb sűrű nemmerev regisztráció egy átlagreferenciához, aztán a „térben változó elmosódás” kezelése, végül dekonvolúció.
- **Diffeomorf template-regisztráció** (Lao és mtsai., 2024): kulcskocka, optikai áramlás a többi képkockára, a warpok átlagolása, ebből torzításmentes referencia. **Fontos:** a sima átlagkép referencia maga is torzult/elmosott, ezt korrigálja.
- **Fizikai alapú warp-prior** (arXiv 1905.07498): a turbulens torzítás nem tetszőleges, hanem fázisernyők hatása. Sima, korlátos sávszélességű eltolási mezővel modellezhető.

---

## 2. Hibakatalógus: minden hibaforrás és a kezelése

Ez a lényeg. A telefonos, afokális, nem követő felvételnél **sokkal több** a geometriai és fotometriai hiba, mint egy dedikált bolygókameránál. Az AS!3 ezek egy részére nincs felkészítve.

| # | Jelenség | Ok | Kezelés |
|---|---|---|---|
| 1 | **Vándorlás** a látómezőn át | A Dobson nem követ | Durva követés (korong-szegmentálás, limbusív-illesztés, súlypont) **Kalman-szűrővel** (közel lineáris drift). Erre finom NCC/fázis-korreláció. Képkockánként érvényességi maszk. |
| 2 | **A Hold részben kilóg** | Drift | **Nem** a közös metszet vágása, mint a PSS-nél, hanem pixelenkénti súlytérkép (Σsúly). Ahol kevés a minta, ott jelezni kell. |
| 3 | **Billegés, elfordulás** | **Mezőrotáció** (alt-az), telefon mozgása az adapterben | A mezőrotáció sebessége ≈ 15,04°/h · cos(φ) · cos(A) / cos(h). φ=47°, déli irány, h=60° esetén **≈0,34°/perc**, ez 1500 px-es Holdon a peremen **≈4–5 px/perc**. **Nem elhanyagolható.** A globális modell tehát legalább **hasonlósági** (eltolás + forgatás + skála): ECC vagy Fourier–Mellin. |
| 4 | **Lüktetés, pulzálás (méret)** | Autofókusz-vadászat (*focus breathing*: a fókusz a skálát is változtatja), digitális stabilizálás (EIS) crop/zoom | Felvételkor AF-zár és EIS kikapcsolása. Feldolgozáskor képkockánkénti skála-becslés (hasonlósági modell). A skála-kiugró képkockák elvetése vagy átméretezése. Az AF-vadászat közbeni képkockák minőségi pontszáma is alacsony lesz. |
| 5 | **Lüktetés, pulzálás (fényesség)** | Auto-expozíció/ISO, átlátszóság, vékony felhő | AE-zár. Feldolgozáskor fényesség-normált minőségmérték (Siril). Stack előtt képkockánként erősítés+offset illesztés a referenciához, robusztusan, a fényes, nem telített pixeleken. Zero-mean NCC. |
| 6 | **Hullámzás** (helyi torzulás) | Seeing (anizoplanatikus turbulencia) | AP-rács, lokális NCC büntetőtaggal, **shift-simítás** a szomszédok között, **sűrű warp-mező** (thin-plate/bicubic interpoláció), AP-nkénti lucky választás, finomító menet. |
| 7 | **Jello / ferdülés** | **Rolling shutter:** a sorok eltérő időben exponálnak, a rezgés soronként eltérő eltolást okoz | A globális modell ezt nem tudja. A sűrű AP-warp korrigálja, ha az AP-k sűrűsége y-irányban elég (≤ 50 px). Opcionálisan sávonkénti (sor-blokk) eltolásmodell. |
| 8 | **Nem tengelyes, ferde telefon: vinyettálás** | A telefon lencséje nincs a kilépő pupillában | Aszimmetrikus, **szenzorhoz rögzített** erősítés-mező G(x,y). A Hold végigvándorol a szenzoron, így **ugyanaz a kráter különböző helyeken** látszik. Ebből a vinyettálás magából a videóból becsülhető (önkalibráló flat): alacsony fokú 2D polinom, robusztus legkisebb négyzetek. Utána minden képkocka osztása G-vel. |
| 9 | **Nem tengelyes, ferde telefon: geometriai torzítás** | Ferde szenzor (trapéz/perspektíva), okulár és telefonlencse torzítása (hordó/párna), mezőfüggő | A torzítás **szenzorhoz rögzített**. A Hold driftje miatt minden képkockában a Hold más részét torzítja másképp, ezért egyetlen hasonlósági transzformáció **nem elég**. Két megoldás: (a) **homográfia** és a maradékot az AP-warp viszi; (b) **önkalibráció:** a drift során a merev Hold elmozdulásaiból a szenzorhoz kötött torzítási mező (radiális + tangenciális + projektív) becsülhető. Ez után a képkockák „kiegyenesítve” mennek a stackbe. |
| 10 | **Színes szegély, a csatornák eltolódása** | Laterális kromatikus aberráció tengelyen kívül, légköri diszperzió | RGB-igazítás. Mivel ez is mezőfüggő, **AP-nként** vagy legalább zónánként, nem csak globálisan. |
| 11 | **Mozgási elmosódás** | Rezgés (a Dobson érintése, szél) az expozíció alatt | A minőségmérték kiszűri. Irány-érzékeny mérték is lehet (Sobel x/y arány). |
| 12 | **Tömörítési hibák** | H.264/HEVC 8 bit, makroblokkok, a telefon zajszűrése és túlélesítése | A Laplace-mérték hajlamos a túlélesített vagy blokkos képkockát „élesnek” látni. Ezért noise-robust (Gauss utáni) mérték kell, sávszűrt tartományban. Ha lehet, 10 bites HEVC vagy RAW videó. I-frame-ek jelölése (minőségük eltér). |
| 13 | **Változó képkockasebesség (VFR)** | A telefonok VFR-t írnak | Időbélyegek kiolvasása (ffprobe/PyAV). A Kalman-predikció valós idővel számol, nem képkocka-indexszel. |
| 14 | **Túlexponálás** | Telített területek | Maszkolás az illesztésnél és a minőségmérésnél. A telített AP kimarad. |
| 15 | **Limbus és terminátor** | Az AP félig a sötét égen | A PSS megoldása: a dim_fraction alapján az AP befelé tolódik. A terminátor mentén a legjobb a struktúra, ott sűrűbb AP-k. |

**Felvételi oldal** (ezzel a legtöbb hiba megelőzhető):
- Pro mód, AF/AE/WB zár.
- **EIS kikapcsolva.** Az EIS soronként warpol, és ez tönkreteszi a pixelpontosságot.
- 4K, a legmagasabb bitráta.
- Rögzített telefonadapter, a telefon lencséje a kilépő pupillában.

---

## 3. Javasolt feldolgozási lánc (rétegek)

```
L0  Bemenet-ellenőrzés   ffprobe: felbontás, bitmélység, VFR, időbélyegek, kodek
L1  Dekódolás            float32, luminancia + RGB, telítettségi maszk
L2  Durva követés        korong/limbusív (RANSAC körillesztés), súlypont, Kalman
L3  Globális geometria   hasonlósági (ECC, piramisos) -> homográfia; RANSAC/ORB tartalék
L4  Önkalibráció         szenzorhoz kötött vinyettálás G(x,y) + torzítási mező (opcionális, iteratív)
L5  Minőség              noise-robust Laplace / fényesség, érvényes területen; globális rangsor
L6  Referencia           legjobb 5-10% átlaga -> (diffeomorf) torzításmentesítés -> finomító menet
L7  AP-rács              lépcsőzetes, többméretű (pl. 32/64/128), fényesség/kontraszt/struktúra küszöb
L8  Lokális shift        többszintű NCC + büntetőtag + szubpixel (parabola vagy upsampled FFT),
                         sikertelen mérés jelölése, szomszéd-simítás, kiugrók kiszűrése
L9  Sűrű warp-mező       AP-shiftek -> sima eltolási mező (thin-plate / bicubic) -> cv2.remap
L10 AP-nkénti választás  legjobb 20% AP-nként, lágy (cosinus) határral
L11 Fotometria           képkockánkénti erősítés/offset, vinyettálás-osztás
L12 Stack                súlyozott átlag + sigma-clip + érvényességi maszk; opcionális drizzle 1.5-2x
L13 Utófeldolgozás       lokális RGB-igazítás, wavelet (PSS/Registax képlet), Richardson-Lucy, opc. AI
```

**Pixelpontosság mérése:**
- **Szintetikus tesztvideó-generátor** kell: egy nagy felbontású Hold-képből az összes hibát szimuláljuk (drift, forgás, skálapulzálás, vinyettálás, perspektíva, rolling shutter, seeing-warp, tömörítés).
- Így **ismert a ground truth**, és mérhető a regisztrációs hiba (px RMS). Enélkül a „pixelpontos” állítás nem bizonyítható.

---

## 3b. Mérés a valódi videón: `Video/h001.mp4`

Eszközök: `tools/analyze_video.py`, `tools/analyze_geometry.py`. Eredmények: `elemzes/h001/`.

**Formátum:**
- H.264 High, 8 bit 4:2:0, 3840×2160, ~49 Mbit/s.
- Kijelzési mátrix −90°, ezért a videó álló: 2160×3840.
- 620 képkocka, 21,06 s.
- **VFR:** 11 kiesett képkocka, a leghosszabb szünet 100 ms.
- 21 I-frame, kb. 30 képkockánként.

**Hold és fotometria:**
- Átmérő ~1700 px, dús hold, végig a látómezőben.
- Telítés nincs. Az átlagfényesség szórása 0,56 (8 bites skálán), tehát az AE stabil volt.

**Drift:**
- Összesen ~285 px (dx −149, dy +244), medián 36 px/s.
- **Rezgéses ugrás akár 15 px képkockánként** (max 306 px/s).

**Affin transzformáció a legjobb képkockához (ECC, korreláció ≥ 0,998):**

| | Teljes tartomány | Lassú rész | Gyors rész (rezgés) |
|---|---|---|---|
| skála x | 0,992…1,015 (≈40 px a Holdon) | 0,0049 | 0,0009 (~1,5 px) |
| skála y | 0,986…1,034 (≈80 px) | 0,0116 | 0,0006 (~1 px) |
| forgás | −0,72…+0,37° (≈16 px a peremen), −1,9°/perc trend | 0,22° | 0,057° (~0,85 px) |
| nyírás | ≈52 px a Holdon | | |

**Fő megállapítás: a „lüktetés” és a „billegés” nem időbeli zaj, hanem látómező-függő optikai torzítás.**
- A skála, a nyírás és a forgás 0,93–0,99-es korrelációban követi a Hold helyét a szenzoron.
- Ahogy a Hold 220 px-t lefelé vándorol, a függőleges mérete 4,3%-kal nő.
- Ez a ferde vagy nem tengelyes telefon és az okulár torzítása, vagyis a 2. fejezet 9. hibája.
- A rolling shutter **nem** jelentős: a torzítás nem korrelál a sebességgel.
- A mért −1,9°/perc forgás nagyobb a várt mezőrotációnál, tehát főleg ez is torzítás.

**Következmények:**
1. Egyetlen képkockánkénti affin vagy hasonlósági transzformáció **nem elég** a pixelpontossághoz. Szenzorhoz kötött torzítási mező kell. Ez önkalibrációval becsülhető, mert a merev Hold 285 px-t vándorol.
2. **Vinyettálás:** a fényesség lefelé haladva csökken (korreláció −0,78, ~1,7%). Önkalibráló flat kell.
3. **Az élesség is helyfüggő** (korreláció −0,59 a pozícióval). A globális legjobb 20% képkockái 5 mp-es sávokra bontva: 54 / 49 / 18 / 3. A minőségválogatás tehát **a helyzet miatt torzított**. Az élességet a pozícióhoz képest kell normálni, vagy AP-nként kell választani.
4. A rezgéses ugrások (max 15 px) miatt a globális keresési ablak legyen ≥ 40 px, és Kalman-predikció kell.
5. A VFR miatt a predikció valós időbélyegekkel számoljon.

---

## 3c. Mérés az önálló fotókon: `Teszt_kepek/` (17 JPG)

Eszköz: `tools/analyze_stills.py`. Eredmények: `elemzes/Teszt_kepek/`.

**Felvétel (EXIF):**
- Samsung SM-A336B, 4000×3000, EXIF Orientation=6, tehát álló kép.
- 1/120 s, ISO 50, rögzített expozíció.
- **Digitális zoom 1,72×**, vagyis a kép interpolált, a valós felbontás kb. 2330×1740 px.
- 17 kép 21 s alatt, 1–4 s-os közökkel.

**Geometria a legélesebb képhez** (AKAZE + RANSAC, majd ECC affin):
- Skála 0,993…1,015, forgás −0,26…+0,47°, nyírás ±0,012.
- **Mind 0,93–0,98-as korrelációban követi a Hold helyét.** Ugyanaz a látómező-függő torzítás, mint a videóban.
- A homográfia perspektív része a Hold szélén 10–225 px eltérést jelezne. Ez durva becslés, de mutatja, hogy a merev és az affin modell nem elég.

**Fotometria:**
- **Erős vinyettálás:** a Hold átlagfényessége 164 → 148, ahogy vándorol (korreláció a hellyel **0,99**, ~10%).
- Az élesség a hellyel −0,5-ös korrelációban áll.

**Ami a fotóknál más, mint a videónál:**

| Kérdés | Videó | Fotósorozat | Következmény |
|---|---|---|---|
| Darabszám | 300–1800 | 10–50 | Nem fix 20% kell, hanem **adaptív** választás: minimum darabszám, minőségi küszöb és súlyozás. 17 kép 20%-a csak 3 kép. |
| Elmozdulás két kép között | ≤ 15 px | akár több száz px és nagyobb forgás (a gombnyomás rázza a telefont) | **Jellemzőpontos durva illesztés** (AKAZE/ORB + RANSAC) kell a korreláció előtt. Időbeli predikció (Kalman) itt nem használható. |
| Képfeldolgozás | Egységes kódoló | **Minden képet külön dolgoz fel** a telefon: tónusgörbe, élesítés, zajszűrés | Képenként tónus-illesztés (hisztogram- vagy görbe-illesztés, nem csak erősítés+offset). Az élesítési halók miatt a minőségmérés zajtűrő legyen. |
| Felbontás | 4K, nincs digitális zoom | Digitális zoom 1,72× | Felvételi tanács: 1× (optikai) zoom, a nagyítást a program végezze a stack után. Ha a telefon tud RAW-t, azt érdemes használni. |
| Látómező-blende | | **Az okulár látómező-széle és a halvány égi háttér látszik** (a sarkokban ívelt szél) | Ez a **szenzorhoz rögzített** kép, ami **ellenkezik** a Hold mozgásával, és az illesztést az identitás felé húzza. Maszkolni kell: a Holdat szegmentáljuk, a hátteret és a blendeszélt kizárjuk az illesztésből és a minőségmérésből. (Az Otsu-szegmentálás most a blendén belüli egész világos mezőt fogta meg, ezért a „d=” méretek a CSV-ben nem a Hold méretei.) |
| Időbélyeg | Képkocka-időbélyeg | EXIF DateTimeOriginal + SubsecTime | A sorrend és az időköz innen jön. |

**Követelmény:**
- A programnak egységes bemeneti réteg kell: **videó** (mp4/mov/avi/ser) vagy **képsorozat** (jpg/png/tif/dng, EXIF-forgatással).
- Ugyanaz az igazító motor fut mindkettőn.
- Különbség csak a durva illesztésben (videónál predikció + korreláció, fotónál jellemzőpontok + RANSAC) és a kiválasztási szabályban van.

---

## 4. Az AI szerepe

| Hol | Mit | Megjegyzés |
|---|---|---|
| Regisztráció | Optikai áramlás neuronhálóval (pl. RAFT) a sűrű warp-mezőhöz | Alternatíva vagy ellenőrzés az AP-módszerhez. Lassabb, GPU-t kíván. |
| Minőség | Tanult élesség-pontozó | A Laplace-mérték általában elég. Tömörített telefonvideónál jól jöhet, mert a túlélesítést meg tudja különböztetni a valódi részlettől. |
| Utófeldolgozás | Turbulencia-csökkentő és deblur hálók (pl. DATUM, NAFNet) | **Hallucinációs kockázat:** kitalált kráterrészletek. Csak opcionális, enyhe, és mindig marad mellette a klasszikus eredmény. |

---

## 5. Nyelv és architektúra

**Python** (NumPy, OpenCV, SciPy, PyAV), a forró ciklusok **Numbával** vagy később **Rusttal (PyO3)**.
- A PSS bizonyítja, hogy Pythonban AS!3-szintű minőség érhető el.
- A forró pont az AP × képkocka NCC: több ezer AP × több száz képkocka. Ez Numbával vagy OpenCV `matchTemplate`-tel gyors, szükség esetén Rustba költöztethető.
- GUI később PySide6-tal. Először CLI és a szintetikus teszt-harness.

---

## Források
- [PlanetarySystemStacker (GitHub)](https://github.com/Rolf-Hempel/PlanetarySystemStacker), algoritmus-összefoglaló: `Documentation/algorithm_summary.pdf`
- [Siril – Multi-Point Registration and Stacking](https://siril.readthedocs.io/en/latest/preprocessing/multipoint.html)
- [AutoStakkert! 2 kézikönyv](https://www.astrokraai.nl/software/manual/as2_planet.html)
- [AutoStakkert lucky imaging – Hold](https://astrophotography.pekdar.net/en/autostakkert-lucky-imaging-moon-stacking/)
- [Sharper Moon Shots with AutoStakkert](https://www.photoartfromscience.com/single-post/2017/11/05/sharper-moon-shots-with-autostakkert)
- [Image Reconstruction through Anisoplanatic Turbulence (arXiv 2009.00071)](https://arxiv.org/pdf/2009.00071)
- [Zhu & Milanfar – turbulencia, nemmerev regisztráció](https://users.soe.ucsc.edu/%7emilanfar/publications/journal/manuscript_turbulence.pdf)
- [Micheli, Lou, Soatto, Bertozzi – Linear systems approach](https://www.math.ucla.edu/~bertozzi/papers/Mario-Turbulence-Final.pdf)
- [Diffeomorphic template registration (arXiv 2405.03662)](https://arxiv.org/html/2405.03662v2)
- [Physics-based turbulence restoration (arXiv 1905.07498)](https://pdf.arxiv.org/pdf/1905.07498)
- [Smartphone afokális fotózás, vinyettálás – Tele Vue](https://televue.com/mobile/TV5_page.asp?id=73)
