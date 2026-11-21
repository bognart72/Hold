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

## Nap-mód (fehér fényű napszűrővel készült felvételek)

```powershell
python -m holdstack <videó vagy képmappa> -o eredmeny\nap --target sun
```

A felületen: **Célpont: Nap**.

> ⚠️ Napot fényképezni csak a teleszkóp **elé** szerelt, minősített (ISO 12312-2) napszűrővel szabad. Okulárszűrő, „sun filter” a fókuszban életveszélyes.

A Nap képe a Holdtól lényegesen eltér, ezért a Nap-mód ezeket másképp csinálja:

| Probléma | Megoldás |
|---|---|
| A korong szinte egyenletes, kevés a felszíni részlet | Az élességi rangsor a **perem élességéből** és a felszíni részletből (mértani közép) készül. A felszíni Laplace-mérték egyedül főleg zajt mérne. |
| Kevés jellemzőpont az illesztéshez | Kevés jellemzőpontnál a **korong középpontja és sugara** adja a kezdő becslést. Az ECC **affin** modellel fut, a maszk a **perem sávját** is tartalmazza. |
| **Pici fekete foltok: napfolt vagy porszem?** | **Pormaszk.** A Nap vándorol a szenzoron, a porszem a szenzorhoz rögzített. A program a képkocka és a referencia arányát a *szenzor* koordinátáiban mediánolja: a porszem kiemelkedik, a napfolt (mert mozog) kiesik. Csak a biztos, legalább 3%-os és 6 px-es foltokat korrigálja, máshol a kép érintetlen. |
| Szélsötétedés (limb darkening) | `_nap_kiemelt.png`: a sugárirányú profil kiegyenlítése és enyhe helyi kontrasztkiemelés. A perem menti fáklyamezők is láthatók lesznek. A mérésre a `_stack.tif` való. |
| Ásványi hold | Napnál nem értelmezett, kimarad. |

Pormaszk Holdnál is kérhető: `--dust`. Kikapcsolás: `--no-dust`.

### Mérés szintetikus napfelvételen (ismert „igazsággal”)

`tools/make_sun_test.py` készít tesztfelvételt. Tartalma:
- szélsötétedés, granuláció
- 9 napfolt (umbra 2,5–14 px), fáklyamezők
- **6 szenzorhoz kötött porszem**
- drift, mezőrotáció, szenzorhoz kötött torzítás és vinyettálás
- seeing, változó elmosódás, zaj, narancs szűrőszín, JPEG

`tools/eval_sun.py` ezt méri: részlethűség, foltmélység, hamis foltok.

Nehéz eset (gyenge granuláció 0,6%, kis drift ~85 px, 40 kocka):

| | Hold-mód | **Nap-mód** |
|---|---|---|
| Hamis foltok (porszem nyoma) | 10 db (6–24% mély) | **0** |
| Részlethűség (DoG-korreláció az igazsággal) | 0,59 | **0,88** |
| Napfoltok mélysége, medián | 96,8% | 96,1% |
| Megtalált porszemek | – | **6/6** (hely ±1 px, mélység 25–39%; valódi: 27–43%) |

A legkisebb (2,5 px-es umbrájú) foltok mélységének ~20%-a mindkét módban elvész. Ez a felvételek elmosódásából ered, nem a stackből.

Könnyebb esetben (erős granuláció, ~340 px drift) a Hold-mód is jó (részlethűség 0,95, hamis folt 0), mert a nagy drift magától kiátlagolja a port.

## Grafikus felület

Dupla kattintás a `holdstack_gui.pyw` fájlon, vagy:

```powershell
python -m holdstack.gui
```

A felület részei:
- Bemenet: videófájl vagy képmappa, és a kimeneti mappa tallózással.
- Az összes fontos beállítás. A felület a legutóbbi értékeket megjegyzi (`~/.holdstack_gui.json`).
- **Indítás / Leállítás**, élő napló, szakaszonkénti haladásjelző és eltelt idő.
- **Eredmény betöltése**: egy korábbi futás eredményét jeleníti meg a kimeneti mappából, újrafuttatás nélkül.
- **Hangolás ehhez a felvételhez**: gyors előzetes elemzés, utána a javasolt értékek a mezőkbe kerülnek, az indoklással együtt. Részletek lent.
- **Alapértékek**: visszaállítja a gyári beállításokat.
- A felület a legutóbb használt értékekkel indul (`~/.holdstack_gui.json`).
- Az ablakméret a képernyőhöz igazodik.
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

### Hangolás az adott felvételhez (`--auto`)

```powershell
python -m holdstack Video\h001.mp4 -o eredmeny\h001 --auto
```

```powershell
python -m holdstack.autotune Video\h001.mp4 --json javaslat.json
```

Az első parancs hangolás után a javasolt értékekkel fut. A második csak a javaslatot írja ki, és elmenti JSON-ba.

A hangolás gyors előzetes elemzés: videónál ~80 mintakocka, fotóknál mind. Videónál kb. 1 perc. A mért jellemzőkből ezeket a paramétereket állítja be:

| Mért jellemző | Paraméter |
|---|---|
| Hold-átmérő (px) | AP-box ≈ átmérő/52 (16–64 px) |
| helyi eltolások a globális illesztés után (11 kockán, ±40 px kereséssel) | keresési sugár ≈ 1,5 × a legnagyobb eltolás + 4 px |
| élesség-eloszlás, képkockaszám | legjobb %, legalább ennyi kocka, jelöltek % |
| a Hold vándorlása a képen | flat- és torzítás-kalibráció be/ki (1,5%-nál kisebb szórásnál ki) |
| színcsatorna-eltolás (CA) | színszegély-csillapítás 0,3 / 0,5 / 0,7 |
| beégett pixelek a bemeneten | csúcsfény-levágás 0,85 / 0,80 |
| zajszint × a stackelt kockák száma | élesítés: visszafogott / közepes / erős |

A kifejezetten megadott opció a hangolt értéket is felülírja, például `--auto --percent 25`.

**Mért hatás (17 fotó, alap vs. `--auto`):**

| | Alap (box 32, keresés 16, 20%) | Hangolt (box 56, keresés 22, 30%) |
|---|---|---|
| Megbízható AP-mérés | 96,5% | 97,1% |
| Helyi eltolás p95 | 5,12 px | 4,80 px |
| Zaj | 0,0031 | 0,0029 (−6%) |
| Élességmérték | 3,03e-5 | 2,85e-5 (−6%) |

A hangolás itt stabilabb mérést és kevesebb zajt adott, de kissé kisebb élességet. Szemre a két kép szinte azonos. A nagyobb AP-box durvábban követi a helyi torzulást. Ez egy szabályalapú első becslés.

**Döntés (2026-11-19): a fejlesztés a próbafuttatásos kalibrációval folytatódik.** Lásd lent.

### Próbafuttatásos kalibráció (`--trial`, `--trial-only`)

```powershell
python -m holdstack Video\h001.mp4 -o eredmeny\h001 --trial
```

```powershell
python -m holdstack Teszt_kepek -o eredmeny\kal --trial-only
```

Az első parancs kalibrál, majd a legjobb beállítással elvégzi a teljes feldolgozást. A második csak kalibrál, és az eredményt a `<név>_kalibracio.json` fájlba menti.

A felületen:
- **„Kalibráció próbafuttatással”** gomb: csak kalibrál, a legjobb értéket beírja a mezőkbe, és táblázatban mutatja a próbákat.
- **„Próbafuttatásos kalibráció a futás elején”** jelölőnégyzet: a teljes futás előtt kalibrál.

Működése:
1. A közös, drága lépések egyszer futnak: elemzés, globális illesztés, flat, CA, torzítás.
2. A program kiválaszt egy erősen strukturált, 1024×1024 px-es csempét a Holdból. Ezt legfeljebb 80 képkockán (időben egyenletesen) globálisan illesztve kivágja, és a memóriában tartja.
3. Minden jelölt beállítás valódi mini-stacket kap: AP-mérés, AP-nkénti kiválasztás, warp-mező. A jelöltek:
   - AP-box: 48 / 64 / 96 px, nagy Holdnál 128 px is.
   - Kiválasztási arány videónál 10 / 20 / 35%, fotóknál 30 / 50 / 80%. A kiinduló érték mindig szerepel.
4. Pontszám = középfrekvenciás részlet ÷ √zaj:
   - részlet = var(G(σ1) − G(σ3)) / átlag²
   - zaj = robusztus szórás a legmagasabb frekvencián
5. **Kiterjesztés:** ha a győztes a jelöltlista szélén van (legkisebb vagy legnagyobb box, illetve arány), a program abba az irányba tovább próbál, legfeljebb két körben.
6. A legjobb pontszámú beállítással folytatódik a futás. A jelentés tartalmazza a teljes próbatáblát és a kiinduló beállításhoz mért javulást.

**Eredmény a 17 fotón** (16 + 1 próba, kb. 30 s):

| AP-box | 20% | 30% | 50% | 80% |
|---|---|---|---|---|
| 32 px (kiterjesztés) | 0,0137 | | | |
| **48 px** | **0,0138** | 0,0137 | 0,0131 | 0,0119 |
| 64 px (alap) | 0,0137 | 0,0136 | 0,0131 | 0,0118 |
| 96 px | 0,0135 | 0,0134 | 0,0129 | 0,0117 |
| 128 px | 0,0134 | 0,0133 | 0,0129 | 0,0117 |

- A box mérete keveset számít. A kiválasztási arány sokat: 80%-nál ~15%-kal gyengébb, mert a gyengébb képek elkenik a részletet.
- A teljes képen mérve a kalibrált stack részlete +0,9% az alapértékeshez képest, a zaj azonos. A fotóknál az alapértékek már majdnem optimálisak voltak.
- A szabályalapú `--auto` (112 px-es box, 30%) a leggyengébb volt (részlet −2,5%). A próbafuttatás tehát a megbízhatóbb módszer.

**Eredmény a videón** (620 kocka, 80 kalibrációs kocka, 9 + 2 próba, ~1,5 perc):

| AP-box | 10% | 20% | 35% |
|---|---|---|---|
| **24 px** (2. kiterjesztés) | | **0,0200** | |
| 32 px (1. kiterjesztés) | | 0,0199 | |
| 48 px | 0,0193 | 0,0196 | 0,0194 |
| 64 px (alap) | 0,0191 | 0,0193 | 0,0193 |
| 96 px | 0,0185 | 0,0189 | 0,0189 |

A teljes képen mérve, alap (64 px, 20%) és kalibrált (24 px, 20%) összevetésben:

| | Alap | Kalibrált |
|---|---|---|
| Részlet | 3,514e-4 | 3,576e-4 (+1,8%) |
| Zaj | 0,0013 | 0,0012 (−8%) |
| Élességmérték | 1,287e-4 | 1,332e-4 (+3,5%) |
| Beégés | 0% | 0% |
| AP-k száma | 877 | 4195 |
| Futásidő | 796 s | 1672 s |

- A videón a seeing apró léptékben torzít, ezt a kis AP-box jobban követi. A kiterjesztés itt kellett: a 24 px-es box a kiinduló listában nem is szerepelt.
- **Ára:** ötször annyi AP. Ezért készült a v0.5 (lent).

## v0.5 – minőség az első, és gyorsabb sűrű AP-rács

**Elv: a képminőség az elsődleges szempont, a sebesség csak utána.** Gyorsítás csak akkor kerül be, ha a minőségi mutatók nem romlanak.

| Változás | Hatás |
|---|---|
| AP-nkénti élességmérés integrálképpel | 1,01 → 0,065 s/kocka (4195 AP), azonos eredmény (eltérés 4e-7) |
| AP-mérés külön folyamatokban (GIL nélkül) | azonos számítás, valódi párhuzamosság |
| Torzítás sűrű táblázatból (1 px-es rács, bilineáris) | 1,75 s → ~0,02 s/kocka, hiba < 0,001 px |
| Eltolási mező rácsa = AP-lépés/4 (2–8 px), bilineáris szétosztás | mintavételi hiba < ~0,02 px |
| **Visszaállítva minőségért:** ECC finom szint 30 iteráció (12 helyett) | kisebb globális maradék (fotó p95: 4,87 → 4,50 px) |
| **Visszaállítva minőségért:** torzítás-kalibráció 60 kocka, 600 pont (30/400 helyett) | videó: torzítás-maradék 0,518 → 0,500 px |
| Kipróbálva és elvetve: kötegelt FFT-s AP-mérés | lassabb volt (1,85 s vs 0,46 s), lapos csúcsoknál eltérő csúcsot adott |

**Videó, kalibrált futás (24 px-es box, 20%):**

| | v0.4 (tegnap) | v0.5 |
|---|---|---|
| Részlet | 3,5764e-4 | **3,5968e-4 (+0,6%)** |
| Élességmérték | 1,3319e-4 | **1,3428e-4 (+0,8%)** |
| Zaj | 0,00123 | 0,00123 |
| Torzítás-maradék | 0,518 px | **0,500 px** |
| Futásidő | 1672 s | **1290 s (−23%)** |

Az alapértékes v0.3-hoz képest (64 px, kalibráció nélkül): részlet +2,4%, élesség +4,3%, zaj −2%.

A mostani leghosszabb lépések a minőségi beállítások miatt vannak:
- torzítás-illesztés 60 kockán: ~270 s
- globális ECC 30 iterációval: ~230 s

Példa: a mintavideóra 36 px-es AP-boxot, 13 px-es keresést, erős élesítést és 0,7-es színszegély-csillapítást javasol. A 17 fotóra 56 px-es boxot (a digitális zoom miatt ~2980 px a Hold), 22 px-es keresést, 30%-ot és legalább 5 képet.

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

## v0.3 – gyorsítás

| Bemenet | v0.2 | v0.3 | Minőség |
|---|---|---|---|
| Videó, 620 kocka, 4K | 1483 s (~25 perc) | **796 s (~13 perc)** | élesség 1,293e-4 → 1,287e-4, fényesség azonos, torzítás-maradék 0,50 → 0,52 px |
| 17 fotó, 12 MP | 249 s | **141 s** | élesség 3,025e-5 → 3,030e-5, fényesség azonos |

A gyorsítás forrásai:
- **ROI munkaterület:** az első referencia után minden mérés és a stack csak a Hold körüli kivágáson fut. A videón ez a kép 55%-a.
- **Durva rácsú leképezés:** az eltolási mező, a homográfia, a torzítás és a CA 8 px-es rácson számolódik, majd lineárisan felskálázódik. A hiba < 0,01 px.
- **CA a stack remapjébe olvasztva:** csatornánként egyetlen interpoláció, nincs külön teljes képes CA-lépés.
- **Gyors elemzés:** szürke uint8, élesség és telítés csak a Hold kivágásán. 207 s → 55 s.
- **Rövidebb ECC finom szint, és a torzítás-kalibráció 30 kockán:** a maradékot úgyis az AP-mérés veszi át.

A videón a kromatikus aberráció érdemi volt: R legfeljebb 1,06 px, B legfeljebb 0,68 px. Ezt a program képkockánként korrigálta.

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
