# Scripts index

Analysis code for all reported results. Absolute paths were anonymized
to `/Users/<ANON>` for blind review. Each script is seeded and writes a
self-contained research note; see the paper's build notes (build08.md-
build14.md) and `../manifests/decision_rules.csv` for the frozen rules.

| script | role | sha256 (first 16) |
|---|---|---|
| `build02_lib.py` | canonical L-profile + hybrid factorizer + value extraction | `93416134abff2153` |
| `build02_rescue.py` | Build 02 rescue pass | `ae88b02b5299ec46` |
| `build09_graphs_v2.py` | graph-automorphism helper (hunt09) | `9c7f9d4887554ed5` |
| `build09_rescue.py` | Build 09 rescue pass | `abd487c77afa6896` |
| `common.py` | shared constants/helpers (paths anonymized) | `98e476ad8d80b707` |
| `stage10_build04.py` | supervised domain signal + overshoot (Build 04) | `15bb034cd1e7d971` |
| `stage11_build05.py` | unsupervised topology, intrinsic dimension (Build 05; Sec 3) | `2363664fdabc7fa7` |
| `stage12_build06.py` | deep-strip + scale-clean rescoring (Build 06) | `684ab120f009009c` |
| `stage13_build06a.py` | resonance / residue-quantization driver (Build 06A; Sec 3) | `cb2866b322c48cb5` |
| `stage14_gate07.py` | synthetic multiplication-detector gate (gate07; Sec 7) | `462105142b5da4ce` |
| `stage15_hunt08.py` | structured-product hunt, closed-form factorizations (hunt08; Sec 7) | `53c12aa7c5bb9ce8` |
| `stage16_hunt09.py` | COD/NIST/graph channels + residual verdict (hunt09; Sec 4,7) | `f7035df9d78a825f` |
| `stage17_hard10.py` | semiprime battery, spike-in, binary grid, degradation, integrity (hard10; Sec 4,5,6) | `4f699df6292a1a41` |
| `stage18_emp11.py` | employment survey-vs-census two-axis contrast (emp11; Sec 6) | `89e6b172f8983f5a` |
| `stage19_whisp12.py` | whisper replicate-or-retire; ACS provenance (whisp12; Sec 6, misfire) | `cd4e97b65aab9a39` |
| `stage1_inventory.py` | corpus inventory (Build 01) | `9013941b8b48ccae` |
| `stage20_power13.py` | quotient-level de-circularized spike-in + naive benchmark (power13; Sec 5, Table 1) | `43eaa35184728d32` |
| `stage21_vote14.py` | election-returns provenance benchmark + Beber-Scacco (vote14; Sec 6, Table 1) | `9e1e7bd9d16b22b8` |
| `stage2_classify.py` | channel classification (Build 01) | `f4c5c1e3fd1e56f2` |
| `stage3_profile.py` | channel L-profiles (Build 01) | `bb5ab34172d9afed` |
| `stage3b_rescue.py` | profiling rescue pass | `a676e2f88b7c8f8c` |
| `stage4_catalog.py` | catalog assembly (Build 01) | `e6710f7629c0be57` |
| `stage5_diag01b.py` | diagnostics (Build 01b) | `f0becdfd2c062961` |
| `stage6_freeze01c.py` | corpus freeze (Build 01c) | `5de16a3d3a614bf3` |
| `stage7_resolve01d.py` | domain/family resolution (Build 01d; paths anonymized) | `e0803ba199cf9793` |
| `stage8_build02.py` | baseline + corpus factorization (Build 02) | `0be7e975b1345a51` |
| `stage9_build03.py` | de-rounding decomposition + clustering teardown (Build 03; Sec 3) | `d667e99b47d7a020` |
| `stage22_geo15.py` | L1/L3 geography disposition, Blocks A--D (Build geo15, **exploratory**; v10 Secs 4--6) | `596c9c0714d20b19` |
| `test_geo15.py` | unit tests for stage22_geo15 (structural zeros, transforms, residuals) | `ee0dee88579c9723` |

## scripts_v10/ — main-text figure scripts (v10 manuscript)

Each script regenerates one main-text figure from the bundled tables
(deterministic; style in `figstyle.py`). `fig6_exact.py` additionally reads
`frozen/observational_corpus_v2.csv` from the private analysis tree (hash in
the freeze manifest) to restrict the recorded-family row to the 367
frozen-corpus families.

| script | figure | hash16 |
|---|---|---|
| `scripts_v10/figstyle.py` | shared style (paths anonymized) | `4a991258d96e5e79` |
| `scripts_v10/fig1_cvap.py` | Fig 1 CVAP false alarm + five-grid attribution | `f595d0e92f2e91b6` |
| `scripts_v10/fig2_geography.py` | Fig 2 raw vs controlled (L1,L3) geography | `6192b1fa178fd4f8` |
| `scripts_v10/fig3_waterfall.py` | Fig 3 four-driver removal + residue controls | `6493acdc773eb58d` |
| `scripts_v10/fig4_mechanism_power.py` | Fig 4 destruction + quotient spike-in power | `559d66fc7c45cfa1` |
| `scripts_v10/fig5_acs.py` | Fig 5 ACS margin-of-error localization | `1a2efb4c6ba5feb2` |
| `scripts_v10/fig6_exact.py` | Fig 6 exact-integer positive contrast | `26ed85b2e494801d` |

## v10.3 exploratory stages (review round 1; plans frozen before results)

| script | role | hash16 |
|---|---|---|
| `stage23_loso.py` | TEST 1: leave-one-source-out robustness of the reduction + core collapse | `b2f0f7226e43af49` |
| `stage24_taskpos.py` | TEST 2: task-sensitivity positive control + S1-stage fold diagnostics | `c09e40293fdd1053` |
| `stage25_flatness.py` | TEST 3: local-flatness stress (failure regimes) | `9c6bcd89a1f72b91` |
| `stage26_counthosts.py` | TEST 4: count-kind quotient spike-in hosts | `0e1d99333d3a9cda` |
| `stage27_gate.py` | TESTS 5+7: co-occurrence example + ground-truth gate library | `b5b2758e6b38c4af` |
| `stage28_powerfp.py` | TEST 6: power curves + FP calibration (assembly from frozen power13) | `15da9b093e9bc63d` |
| `stage29_pairedci.py` | v10.4: paired family-bootstrap CIs for the transfer-ladder deltas | `2d76d0a9afcc30ba` |
