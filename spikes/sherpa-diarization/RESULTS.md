# sherpa-onnx diarizace vs. pyannote community-1 (meeting16)

Datum: 2026-11-12, Apple M5 (4P + 6E jader, 32 GB), sherpa-onnx 2.0.1 (Python bindingy, onnxruntime v balíčku).
Pipeline: segmentace `pyannote-segmentation-3.0` (ONNX, fp32) → speaker embedding → `FastClustering`
(aglomerativní shlukování, buď `num_clusters=5`, nebo práh), `min_duration_on=0.3`, `min_duration_off=0.5`.
Hodnocení: `scripts/assemble.py` beze změny (shoda mluvčích s Teams VTT na úrovni slov, whisper slova jsou stejná jako u pyannote).

„Falešní“ = labely, které se při mapování 1:1 na mluvčí z Teams nenamapovaly; v závorce je jejich řeč v sekundách.
„Teams nezmapován“ = skutečný mluvčí, kterého diarizace sloučila s jiným.

## Výsledky: výřez 33:30–37:30 (240 s, 246 slov, 5 mluvčích, nejmenší má 5 slov)

| Varianta | Shoda | Mluvčích | Falešní (s) | Teams nezmapován |
|---|---|---|---|---|
| **pyannote community-1** | **94 %** | 6 | 2,5 | 0 |
| TitaNet‑L, práh 1.0 | 94 % | 6 | 1,0 | 0 |
| TitaNet‑L, práh 1.05 | 94 % | 6 | 1,0 | 0 |
| TitaNet‑L, práh 1.1 | 74 % | 5 | 1,3 | 1 |
| TitaNet‑L, práh 1.05 + sloučení < 20 s | 93 % | 4 | – | 1 (ten s 5 slovy) |
| TitaNet‑L, `num_clusters=5` | 74 % | 5 | 1,3 | 1 |
| ERes2NetV2 (3D-Speaker), práh 0.9 / 1.0 | 93 % | 5 | – | 0 |
| ERes2NetV2, `num_clusters=5` | 93 % | 5 | – | 0 |
| ERes2Net base (3D-Speaker), práh 1.0 | 92 % | 5 | 1,0 | 1 |
| ERes2Net base, `num_clusters=5` | 92 % | 4 | – | 1 |
| WeSpeaker ResNet34‑LM, práh 0.5 | 81 % | 7 | 4,2 + 0,6 | 0 |
| WeSpeaker ResNet34‑LM, `num_clusters=5` | 64 % | 5 | 4,2 | 2 |
| CAM++ VoxCeleb (3D-Speaker), nejlepší práh 0.8 | 55 % | 9 | 10 | 1 |

Prahy 0.4–0.8 dávají u TitaNet, ERes2Net i CAM++ 10–28 mluvčích; použitelné hodnoty jsou kolem 0.9–1.05 (ResNet34‑LM kolem 0.5).

## Výsledky: celá nahrávka (2 924 s, 3 585 slov, 5 mluvčích + 1 s jedním slovem)

| Varianta | Shoda | Mluvčích | Falešní (s) | Teams nezmapován |
|---|---|---|---|---|
| **pyannote community-1 (odhad)** | **95 %** | 7 | 36, 32 | 0 |
| pyannote, `num_speakers=5` | 91 % | 5 | – | 1 |
| TitaNet‑L, práh 1.0 | 94 % | 28 | 50 + 22 drobných (≤ 7 s) | 0 |
| TitaNet‑L, práh 1.05 | 94 % | 22 | 57 + 16 drobných | 0 |
| TitaNet‑L, práh 1.1 | 96 % | 17 | 12 drobných (≤ 10 s) | 0 |
| TitaNet‑L, práh 1.15 | 97 % | 12 | 7 drobných (≤ 14 s) | 0 |
| TitaNet‑L, práh 1.2 | 92 % | 6 | 14, 9 | 1 |
| TitaNet‑L, práh 1.0 + sloučení < 20 s | 96 % | 6 | 56 | 0 |
| **TitaNet‑L, práh 1.05 + sloučení < 20 s** | **95 %** | 6 | 60 | 0 |
| TitaNet‑L, práh 1.1 + sloučení < 20 s | 97 % | 5 | – | 0 |
| TitaNet‑L, práh 1.0 + sloučení < 60 s (přeladěno) | 98 % | 5 | – | 0 |
| TitaNet‑L, `num_clusters=5` | 92 % | 5 | 14 | 1 |
| ERes2NetV2, práh 0.9 | 86 % | 23 | 119, 105, 24, 20, … | 0 |
| ERes2NetV2, práh 1.0 (+ sloučení < 20 s) | 87 % | 13 (6) | 117, 26 | 0 (1) |
| ERes2NetV2, `num_clusters=5` | 80 % | 5 | 1,2 | 2 |
| ERes2Net base, práh 1.1 (+ sloučení < 20 s) | 88 % (89 %) | 14 (6) | 193 (195) | 0 |
| ERes2Net base, práh 1.0 | 80 % | 31 | 189, 138, … | 0 |
| ERes2Net base, `num_clusters=5` | 84 % | 5 | – | 0 |
| WeSpeaker ResNet34‑LM, práh 0.6 | 65 % | 10 | 117, 94, … | 0 |
| WeSpeaker ResNet34‑LM, práh 0.5 | 53 % | 17 | 238, 220, … | 0 |
| WeSpeaker ResNet34‑LM, `num_clusters=5` | 49 % | 5 | 4 | 1 |

„Sloučení < N s“ je vlastní post-processing (`scripts/merge_small.py`, obdoba `min_cluster_size` v pyannote):
clustery s méně než N s řeči se segment po segmentu přiřadí k nejbližšímu velkému clusteru podle kosinové
podobnosti embeddingu (sherpa-onnx `SpeakerEmbeddingExtractor`, stejný model). V Rustu je to pár desítek řádků nad
stejným C API. Nejmenší skutečný mluvčí má na celé nahrávce zhruba 80 s řeči, takže práh 60 s je vyladěný na tuto
nahrávku a nelze ho brát jako default. Na výřezu by „sloučení“ odstranilo každého, kdo řekne jen pár slov.

## Čas a paměť (TitaNet‑L, práh 1.05, nic jiného neběželo)

| Běh | Výřez 240 s | Celá 2 924 s | Max RSS (celá) |
|---|---|---|---|
| CPU, 4 vlákna | 11,8 s | 166 s (RTF 0,057) | 1,3 GB |
| CPU, 8 vláken | 13,7 s | 198 s | 1,3 GB |
| CoreML EP, 4 vlákna | 27,2 s | 356 s | 3,0 GB |
| + sloučení malých clusterů (CPU) | ~5 s | 45 s | – |
| pyannote community-1 MPS / CPU | 21 s / 165 s | ~230 s (MPS) | – |

- Načtení modelů 0,1 s (CPU), 1,1 s (CoreML).
- CoreML EP v balíčku je, ale je 2× pomalejší a bere 2,3× víc paměti; výsledek je stejný (na celé nahrávce 95 % místo
  94 %, malé rozdíly ve shlukování). Víc vláken než 4 výkonná jádra zpomaluje (E-jádra).
- Ostatní embeddingy na výřezu (CPU, 4 vlákna): CAM++ 8,5 s, ResNet34‑LM 14,7 s, ERes2Net base 15,5 s, ERes2NetV2 30 s.
  Na celé nahrávce běžely tři varianty paralelně (3 vlákna), takže časy jsou jen orientační: ResNet34‑LM ~340 s,
  TitaNet ~370 s, ERes2Net base ~520 s, ERes2NetV2 ~600–900 s.
- Paměť roste s délkou (embeddingy a matice vzdáleností pro všechny segmenty): výřez 0,25–0,8 GB, celá nahrávka
  0,8–1,6 GB podle modelu.
- int8 segmentační model (`model.int8.onnx`) nebyl rychlejší (15 s) a shoda na výřezu klesla na 74 %, takže ho nepoužívat.

## Kde se nejlepší varianta plete (TitaNet‑L, práh 1.05 + sloučení < 20 s)

Mluvčí jsou anonymizovaní písmeny podle počtu promluv v Teams (A = nejvíc); písmena výřezu a celé nahrávky si neodpovídají.

- **Celá nahrávka (173 chyb z 3 585):** hlavní chyba je **rozštěpení** jednoho mluvčího (B, druhý nejaktivnější) na dva
  clustery: 88 slov / 60 s jde do falešného labelu. pyannote dělá stejný typ chyby, jen jinde: mluvčí A má 49 slov ve
  falešném labelu, B 21+6 a D 20. Zbytek jsou záměny mezi skutečnými mluvčími (A↔B 12/12, B→D 10, C→D 7).
  Při prahu 1.1 rozštěpení zmizí a chyb je 91 (97,5 %).
- **Rychlé výměny:** 60 % chyb sherpa (pyannote 57 %) je do 3 slov od změny mluvčího v Teams. Je to vlastnost
  přiřazení slov k segmentům a stejné segmentace, ne embeddingu.
- **Překryvy:** sherpa vrací méně překrývajících se úseků než pyannote. Ve slovech, kde se diarizace překrývá, je 61 chyb
  ze 143 slov (pyannote 59 ze 214).
- **Výřez:** stejná čísla jako pyannote (93–94 %), chyby jsou jinde. Sherpa zaměňuje C→D (8 slov) a mluvčího s 5 slovy
  přiřadí jinam (to dělá i pyannote). pyannote se plete v D→B a C→D.
- **Vynucení 5 mluvčích** selhává stejně jako u pyannote, protože sloučí nejmenšího skutečného mluvčího s jiným
  (E→D 159 slov, shoda 92 %).

## Licence a dostupnost modelů

| Komponenta | Zdroj ke stažení | Gated? | Licence | Poznámka pro distribuci |
|---|---|---|---|---|
| sherpa-onnx (runtime, C API, Rust crate `sherpa-rs`/vlastní FFI) | github.com/k2-fsa/sherpa-onnx | ne | Apache-2.0 | onnxruntime je MIT |
| pyannote segmentation-3.0 (ONNX) | GitHub release `speaker-segmentation-models` (sherpa-onnx) | ne (originál na HF je gated click-through) | MIT (© CNRS, LICENSE je v archivu) | lze přibalit, ponechat copyright notice |
| NeMo TitaNet‑Large (ONNX) | GitHub release `speaker-recongition-models` (originál `nvidia/speakerverification_en_titanet_large`) | ne | CC-BY-4.0 | lze přibalit, nutná atribuce NVIDIA; 97 MB |
| 3D-Speaker ERes2Net base / ERes2NetV2 / CAM++ | stejný GitHub release (originály `iic/…` na ModelScope) | ne | Apache-2.0 | 38 / 68 / 28 MB |
| WeSpeaker ResNet34‑LM | stejný GitHub release (originál `Wespeaker/…` na HF) | ne | CC-BY-4.0 | 25 MB |
| *pro srovnání* pyannote community-1 | HF `pyannote/speaker-diarization-community-1` | **ano** (HF token + souhlas) | CC-BY-4.0 | stáhnout bez tokenu nejde |

Licence jsou ověřené přes HF a ModelScope API a LICENSE v archivu. SHA-256 stažených souborů sedí s `checksum.txt` v releasu.
Žádný z použitých ONNX modelů není gated ani nekomerční. Pozor na `sherpa-onnx-reverb-diarization-v1/v2`
ve stejném releasu: Rev Reverb má podle všeho nekomerční licenci (neověřoval jsem), ten jsem netestoval ani nepoužíval.

## Doporučení

**Kvalita na Rust variantu stačí, ale jen s TitaNet‑Large a s vlastním sloučením malých clusterů.**

- Nejlepší robustní nastavení: **segmentation-3.0 (fp32) + NeMo TitaNet‑Large, `FastClustering` práh 1.05,
  pak sloučení clusterů < 20 s do nejbližšího velkého**. Výřez 93 %, celá nahrávka 95 % (pyannote 94 % a 95 %).
  Má jednoho falešného mluvčího s 60 s (rozštěpený skutečný mluvčí), pyannote měl dva (36 s a 32 s). Na CPU
  ~0,07 RTF včetně sloučení (celá hodina ~3,5 min), tedy srovnatelně s pyannote na MPS. Bez Pythonu a bez HF tokenu.
- Práh je citlivý na délku nahrávky: na celé nahrávce je nejlepší 1.1–1.15 (97 %), na 4min výřezu ale 1.1 slučuje dva
  hlavní mluvčí (74 %). Pro aplikaci proto 1.05 (případně 1.0) + sloučení, s možností ručně zadat počet mluvčích
  nebo „rozdělit/sloučit“ v UI. Vynucení počtu mluvčích dopadá stejně špatně jako u pyannote (slučuje nejmenšího).
- Bez post-processingu sherpa na delší nahrávce vyrábí desítky drobných clusterů (17–28 labelů). Sloučení malých
  clusterů je pro produkt nutné. Stojí ~45 s na hodinu audia, protože Python API neexportuje embeddingy z pipeline;
  v Rustu by šlo embeddingy počítat jednou a celé shlukování dělat sám.
- ERes2NetV2 a ERes2Net base jsou na krátkém výřezu dobré (92–93 %), ale na hodinové nahrávce padají na 84–89 %
  se 2–3minutovými falešnými mluvčími, takže nevyhovují. WeSpeaker ResNet34‑LM (stejný embedding jako v pyannote) a
  CAM++ jsou v jednoduchém shlukování sherpa nepoužitelné (≤ 81 %). Náskok pyannote dělá jeho shlukování (VBx/PLDA),
  ne embedding.
- CoreML EP nepoužívat (pomalejší, víc paměti). CPU se 4 vlákny.
- Omezení měření: jedna schůzka, 5 mluvčích, čeština a slovenština. Práh 1.05 a hranice 20 s jsou z ní vyladěné; před
  rozhodnutím o produktu stojí za to ověřit je na 2–3 dalších nahrávkách (jiný počet mluvčích, jiná délka).

## Soubory

- `scripts/diar.py`: diarizace sherpa-onnx → JSON ve formátu spiku (+ `max_rss_mb`, parametry).
- `scripts/merge_small.py`: sloučení malých clusterů. `scripts/score.py`: wrapper nad `assemble.py`, vypíše řádek bez jmen.
- `scripts/errors.py`: rozbor chyb s anonymizovanými mluvčími. `scripts/sweep.sh`: sweep variant.
- (jen lokálně, ne v repu) `out/*.json`: všechny diarizace, `out/results.tsv`: souhrn, `out/timing/`: izolované časové běhy.
- (jen lokálně, ne v repu) `models/`: stažené modely (jen ONNX a archiv segmentace, skripty z archivu se nespouštěly).
