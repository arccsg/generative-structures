# RESULT_PROVENANCE.md — every numerical/empirical claim retained in v10

Scope: claims retained in `lprofile_paper_v10_recording.tex` (main) and
`lprofile_supplement_v10.tex`. The authoritative machine-readable map is
`artifact/claim_manifest.csv` (36 claims, each with script, output table, config
and SHA-256 of the output); this file adds registration status, seeds, and the
verification performed during the 2026-07-12 restructure. Analysis repo:
`~/Projects/lprofile-geography` (paths below relative to it). Seeds follow the
per-build convention recorded in `config/run_config_*.json`.

Verification legend:
- **re-read** — output CSV re-read on 2026-07-12; value matches the manuscript.
- **hash** — output covered by `artifact/hashes` / `claim_manifest.csv` SHA-256; not independently recomputed now.
- **rerun** — recomputed during this restructure (new exploratory work).

| # | Claim (short) | Script | Data | Config (frozen) | Output | Seed | Registration | Verified |
|---|---|---|---|---|---|---|---|---|
| P1 | CVAP Beber–Scacco χ²=12,684.014, p≈0; all enumerated series p≥0.158 | `scripts/stage21_vote14.py` | CVAP 2022 special tab; NC SBE precinct returns; certified county returns (cached under `diagnostics/intermediate02/emp11_cache/`) | `run_config_vote14.json` | `tables/vote14_beber_scacco.csv` | 20260714 | pre-registered (V14-2) | re-read ✓ |
| P2 | CVAP residue excess Q25 +0.773, Q100 +0.729; enumerated ≈0 | same | same | same | `tables/vote14_grid_probe.csv` | 20260714 | pre-registered (V14-2) | re-read ✓ |
| P3 | CVAP grid mass 0.275 vs votes 0.12; depth 0.55 | same | same | same | `tables/vote14_profiles.csv` | 20260714 | pre-registered (V14-2) | re-read ✓ |
| P4 | Votes multiplicative null k_eff 0.98–1.03; whisper residuals +0.004..+0.016 small-count artifact | same | same | same | `tables/vote14_profiles.csv` | 20260714 | pre-registered (V14-1) | re-read ✓ |
| P5 | Corpus 18,951 channels / 367 families / 18 domains (~9.6e8 factorizations) | `scripts/stage9_build03.py` (+freeze) | frozen v2 corpus | `run_config_03.json` | `frozen/observational_corpus_v2.csv` (sha 03e2c620…), `tables/family_inventory.csv` | 20260702 | frozen corpus | re-read ✓ (row count) |
| P6 | Raw (L1,Tail) three clusters, silhouette 0.719 | `scripts/stage9_build03.py` | frozen corpus | `run_config_03.json` | `tables/discovery_clusters.csv`, `cluster_membership.csv` | 20260704 | exploratory (Build 03), reported as such | hash |
| P7 | Cluster survival Jaccard 0.39–0.43 after 2·5 strip | same | same | same | `tables/cluster_survival.csv` | 20260704 | exploratory | re-read ✓ |
| P8 | Monetary domain signal reverses sign scale-clean (dL2 +0.042 → dL2′ −0.045) | `scripts/stage12_build06.py` | frozen corpus | `run_config_06.json` | `tables/accounting_rescored.csv`, `frozen/shape_features.csv` | 20260706 | exploratory | hash |
| P9 | Cleaned-space intrinsic dimension 1.16 (TwoNN), no domain islands | `scripts/stage11_build05.py` | frozen corpus | `run_config_05.json` | `logs/stage11_build05_discovery.log`, `tables/unsup_vs_domain.csv` | 20260705 | exploratory | log re-read ✓ |
| P10 | Residue-quantization negative controls 10–30× separation (planted resonant 0.051–0.167 vs generic ≈0.0056) | `scripts/stage13_build06a.py` | frozen corpus + planted controls | `run_config_06a.json` | `tables/resonance_controls.csv`, `residue_structure.csv` | 20260706 | pre-registered controls | re-read ✓ (generic mean 0.005570; ledger SHA-256 8799e3b4…, matches claim C08) |
| P11 | Surviving concentration excess −0.0184 → −0.0001 (90% family-bootstrap CI [−0.0004,+0.0004]) under core-magnitude null | `scripts/stage16_hunt09.py` | frozen corpus | `run_config_hunt09.json` | `tables/hunt09_residual_rebaseline.csv`, null `tables/baseline_coremag.csv` | 20260709 | pre-registered (H9-1) | re-read ✓ (table present, 362 families) |
| P12 | Exponent-stacking 1.9% observational vs 100% smooth products | same | same | same | `tables/hunt09_residual_mechanism.csv` | 20260709 | pre-registered (H9-1 mech check) | hash |
| P13 | Binary-aligned c2 excess +0.72 with c5 −0.007; unaligned null | `scripts/stage17_hard10.py` | file-size/allocation channels | `run_config_hard10.json` | `tables/hard10_binarygrid.csv` | 20260710 | pre-registered (H10-4 family) | hash |
| P14 | Grid-coarseness degradation AUC 0.86 (no grid) → 0.75 (grid 2) → chance (≥25–100); n→n+1 AUC 0.4437 | same | synthetic products | same | `tables/hard10_degradation.csv` | 20260710 | pre-registered (H10-3, G2) | re-read ✓ |
| P15 | Quotient spike-in: tier (b) recovered 6/6 at f=0.05 and 0.10, confound-clean 5/6; tier (c) 6/6 at f=0.05, clean 6/6; tier (a) confounded (|z| to 1e7 off-grid) | `scripts/stage20_power13.py` | six amount-kind host channels | `run_config_power13.json` | `tables/power13_quotient_spikein.csv`, `power13_confound_check.csv` | 20260713 | pre-registered (P13-1) | re-read ✓ (tables present; det columns) |
| P16 | 47/43 constructed tick: naive battery ≤0.53, encoding AUC 1.00 | same | constructed re-quantization of real values | same | `tables/power13_naive_benchmark.csv` | 20260713 | pre-registered (P13-4) | re-read ✓ |
| P17 | Employment: k_eff 0.98–1.00 all vintages; survey rounding mass 0.474 vs census 0.102 | `scripts/stage18_emp11.py` | BLS CES (ALFRED vintages), QCEW | `run_config_emp11.json` | `tables/emp11_profiles.csv`, `emp11_contrast.csv` | 20260711 | pre-registered (E11-1, E11-2) | hash |
| P18 | ACS point estimates not grid-marked (rm 0.107 vs 0.112; AUC 0.77 below bar); MOE column rm 0.183 [0.151,0.216] vs 0.112 | `scripts/stage19_whisp12.py` | ACS 5-yr B01003, decennial | `run_config_whisp12.json` | `tables/whisp12_acs.csv`, `whisp12_acs_contrast.csv` | 20260712 | pre-registered expectation (misfired direction reported); MOE localization exploratory-then-CI-confirmed | hash |
| P19 | Whisper retired: no CI-clean replication across 4 systems (W12-1) | same | CES vintages, PEP, SA/NSA, ACS | same | `tables/whisp12_*.csv` | 20260712 | pre-registered — **missed**, reported | hash |
| P20 | Benchmark rows (Table: naive vs encoding by contrast, incl. two-sided fabrication read where last-two-digit drift reaches 1.00) | `stage20/stage21` | as above | power13/vote14 configs | `tables/power13_naive_benchmark.csv`, `vote14_benchmark.csv` | 20260713/14 | pre-registered gate (0.6/0.8 rule) | re-read ✓ |
| P21 | Exact regime: sums/Poisson k_eff=1.00; products AUC 0.86–0.96; imbalance to 90/10; one digit of rounding destroys | `scripts/stage14_gate07.py` | synthetic ground truth | `run_config_gate07.json` | `tables/gate_separation.csv`, `gate_rungs.csv`, `gate_detection_floor.csv` | 20260707 | pre-registered (G1, G2) | hash |
| P22 | Structured products (factorials, binomials, multinomials, group orders; magnitudes to 1e7263) AUC ≥0.85 → 0.889–0.998 | `scripts/stage15_hunt08.py` | closed-form constructions | `run_config_hunt08.json` | `tables/hunt_separation.csv`, `hunt_structured.csv` | 20260708 | pre-registered (H8-1) | hash |
| P23 | COD multiplicities k_eff 0.534±0.091 (n=11,240); NIST degeneracies 1.707±0.557 (n=505); graph |Aut| to 1e17,265 k_eff up to 4.44 | `scripts/stage16_hunt09.py` | COD (CC0), NIST ASD, SNAP | `run_config_hunt09.json` | `tables/hunt09_lowkeff.csv`, `hunt09_highkeff.csv` | 20260709 | pre-registered (H9-2, H9-3) | hash |
| P24 | Balanced pairs: k_eff-blind, full-profile AUC 0.83–0.93 | `scripts/stage17_hard10.py` | synthetic semiprimes | `run_config_hard10.json` | `tables/hard10_semiprime.csv` | 20260710 | pre-registered (H10-1) | hash |
| P25 | 20 frozen rules, 17 met, 3 missed (W12-1, P13-2, P13-3) with mechanical causes | all | — | all configs | `artifact/manifests/decision_rules.csv` | — | the registration itself | re-read ✓ |
| P27 | Real additive negative controls: telemetry byte/packet/flow counts k_eff 0.9958±0.0175 (n=6,399 channels), census counts 0.9957 — quoted as 0.996 (main §9, supp §S7) | `scripts/stage15_hunt08.py` | frozen corpus negative-control channels | `run_config_hunt08.json` | `tables/hunt_negative_controls.csv` (artifact SHA256SUMS line 102: b1f5f3b5…) | 20260708 | pre-registered negative control (Build 08) | re-read ✓ 2026-07-12 |
| P28 | Figure 6 recorded-family summary: family-level k_eff over the 367 frozen-corpus families (median 1.05, IQR [0.995, 1.307]); `family_profiles_v3.csv` restricted by `observational_corpus_v2.csv` membership | `scripts/stage9_build03.py` (+ `scripts_v10/fig6_exact.py` restriction) | frozen corpus | `run_config_03.json` | `tables/family_profiles_v3.csv` (SHA-256 ac10d207…, matches claim C04); `frozen/observational_corpus_v2.csv` (03e2c620…) | 20260702 | frozen corpus aggregate; restriction is presentation-only (v10.1) | rerun ✓ |
| P29 | Ledger hashes for Fig 3 / Table 2 driver tables not in the frozen artifact manifest: `derounding_effect.csv` b9fb2394…, `deepstrip_effect.csv` de310751…, `residue_structure.csv` 878a6d35… (SHA-256, computed 2026-07-12) | `stage9/stage12/stage13` | frozen corpus | build03/06/06a configs | as named | 20260704–06 | frozen driver tables | hashed ✓ |
| P26 | Magnitude-conditioned baseline E[L1|d]: 0.952 (d=1) → 0.633 (d=18); PD limit E[L1]→0.6243 | `scripts/stage8_build02.py` | 200k uniform draws × 18 strata | `run_config_02.json` | `tables/baseline.csv` | 20260702 | frozen null | re-read ✓ (used in Blocks A–B) |

## New exploratory results (this restructure; Build "geo15")

All rows below are **exploratory** (no frozen rule covers them; the ΔAUC≈0.05
retention threshold is a project-management rule, not a preregistered
criterion). Script `scripts/stage22_geo15.py`, config `run_config_geo15.json`,
seed 20260715, outputs `tables/geo15_*.csv`, log `logs/stage22_geo15.log`.
Exact values: see `L1_L3_DISPOSITION.md`.

| # | Claim | Output | Verified |
|---|---|---|---|
| G1 | Raw channel-level (L1,L3) geography and its group structure (nulls / recorded / grids / exact / additive) | `tables/geo15_geography_channel.csv`, fig `figures_v10/fig2_geography_raw_controlled.*` | rerun |
| G2 | Controlled/residual geography after digit-length, c2/c5, modal-grid, deep-strip controls (collapse or survival) | `tables/geo15_controlled.csv` | rerun |
| G3 | Held-out incremental value of L3 beyond recording variables + L1 (grouped by dataset-family; macro OVR AUC; grouped bootstrap CI) | `tables/geo15_heldout_auc.csv` | rerun |
| G4 | Mechanism stress on the same sample (n→n+1, coarse rounding, modal-grid quotient, exact-vs-recorded) | `tables/geo15_stress.csv` | rerun |
| G5 | Per-record vs channel-level observational-level check (structural zeros: records with <3 prime-power parts) | `tables/geo15_perrecord_sample.csv` | rerun |

Rule: no number appears in v10 unless it is in this file or in
`artifact/claim_manifest.csv`. The final report lists any sentence whose number
is not yet tied to an output file.

## v10.3 exploratory extensions (review round 1; plans frozen before results)

All rows exploratory; thresholds are project rules. Scripts in
`~/Projects/lprofile-geography/scripts/`, configs in `config/`, outputs in
`tables/`. Seed labels 20260716–20260721 (labels, not timestamps).

| # | Claim | Script | Config | Output | Seed | Verified |
|---|---|---|---|---|---|---|
| P30 | LOSO robustness: all variants hold the frozen rule; telemetry-excluded core residual −0.0001 [−0.0004,+0.0004]; procurement-excluded −0.0002 [−0.0005,+0.0003]; raw arm +0.091..+0.092, de-round −0.021, strip7 −0.018..−0.019; telemetry/census contribute 0 arm families | `stage23_loso.py` | `run_config_loso23.json` | `loso23_results.csv`, `loso23_family_counts.csv` | 20260716 | rerun ✓ (full variant reproduces frozen −0.0184/−0.0001) |
| P31 | Transfer-design positive control: AUC 0.47–0.49 null → 0.67/0.81 at grid-mass shift +0.020 → 0.99 at +0.066 → 1.00 full grids; tick-7 read at ~1.00; permutation null mean 0.485, range [0.39,0.61] (1/20 draws marginally above the frozen 0.6 bound, reported) | `stage24_taskpos.py` | `run_config_taskpos24.json` | `taskpos24_calibration.csv`, `taskpos24_permutation.csv` | 20260717 | rerun ✓ |
| P32 | geo15 S1-stage diagnostics: fold with 1 test family/1 class (macro-AUC undefined), fold 98% one class (0.864), three balanced folds 0.497–0.509 — fold instability, not negative information | same | same | `taskpos24_s1_diagnostics.csv` | 20260717 | rerun ✓ |
| P33 | Local-flatness failure regimes: quotient dL1 −0.18 / keff 1.34 / c2c5 share 0.64 vs 0.36 at r=0.5; |dL1| ≤ 0.013, keff within 0.02 of 1 at r ≥ 5; ZIP counts dL1 +0.048, keff 0.94; frozen 2SE rule passes only r=5,g=100 (rule severe at N=4e4, stated) | `stage25_flatness.py` | `run_config_flat25.json` | `flat25_results.csv` | 20260718 | rerun ✓ |
| P34 | Count-kind hosts: 3 telemetry byte-count channels (only count channels at frozen magnitude screen); recovery 0/3 at f=0.01, 3/3 at f≥0.05; confound max|z| 3.3–5.2 — NOT clean per frozen rule; all flagged coords are residue scores drifting negative (resonance dilution), decimal-grid coords within noise | `stage26_counthosts.py` | `run_config_count26.json` | `count26_hosts.csv`, `count26_spikein.csv`, `count26_confound.csv` | 20260719 | rerun ✓ |
| P35 | Co-occurrence worked example: fabricated lognormal on five-grid → χ²₉ ≈ 11,792, g0=5, TV excess +0.764@25 → gate reports grid-compatible (attributes the alarm, not the data) | `stage27_gate.py` | `run_config_gate27.json` | `gate27_library.csv` (row FABRICATED_lognormal_5grid) | 20260720 | rerun ✓ |
| P36 | Gate validation library: sensitivity 6/7 (miss: ACS MOE, partial rounding below the 0.05 TV / 0.90 conformance thresholds), specificity 5/6 (false flag: NC precinct, +0.116@mod 8, small-magnitude regime); CES g0=100 TV excess +0.92; 47/43 ticks flagged at own moduli | same | same | `gate27_library.csv` | 20260720 | rerun ✓ |
| P37 | Power curves + FP calibration: per-tier recovery 0.5/1.0/1.0/1.0 (a,b) and 0.0/1.0/1.0/1.0 (c) at f=0.01/0.05/0.10/0.25; FP nominal 0.05 by bootstrap-band construction, single f=0 point below threshold 6/6 hosts | `stage28_powerfp.py` | `run_config_powerfp28.json` | `powerfp28_curves.csv`, `powerfp28_fpcal.csv` | 20260721 | assembled from frozen power13 outputs ✓ |

v10.3 verification notes: Arratia 2000 DOI 10.1214/aop/1019160500 and Buneman
10.1007/3-540-44503-X_20 Crossref-verified correct as printed (reviewer
suspicion unfounded); Abedjan 10.1007/s00778-015-0389-y verified (the -x
variant does not resolve). Fix 0 audit: no empirical claim depended on the
(removed) prime-relabeling invariance statement — every reduction operator
acts on actual prime magnitudes (stripping, valuations), never on relabeled
primes.

## v10.4 additions

| # | Claim | Script | Config | Output | Seed | Verified |
|---|---|---|---|---|---|---|
| P38 | Paired family-bootstrap 90% CIs for the Table S1 ladder deltas (one shared family resample per draw): L1 vs R +0.009 [−0.006,+0.026]; L3 vs R+L1 +0.013 [−0.011,+0.038] (includes 0); full vs R +0.035 [+0.004,+0.073]; full vs R+L1 +0.026 [+0.003,+0.051] | `stage29_pairedci.py` | `run_config_pairedci29.json` (frozen before results) | `pairedci29_deltas.csv` | 20260722 | rerun ✓ (point deltas reproduce geo15 Table S1 pre-rounding values exactly) |

v10.4 numeric reconciliations (definitional, verified from source tables):
- Fig 3(a) re-estimated at the frozen-arm family-equal grain (the §S3/LOSO
  estimand): +0.078→+0.091, −0.015→−0.021, −0.016→−0.018 (the strip bar now
  IS the −0.0184 deep-core statistic). The old values were a
  family-count-weighted domain-level mean over a rounding-mass-defined arm —
  a different weighting/arm set, not an error; the paper now uses one
  estimand everywhere.
- Gate library fractions: sensitivity 6/7 over seven GRID-TRUTH series
  (3 documented real + 4 constructed); §11 "7 documented-grid" corrected.
- k_eff definition: code computes the ratio of averages (build02_lib
  profile_values → E-weighted null over mean H2); §S1 corrected to match
  main Eq. (2).
- Planted-controls separation: 10–30× → ~9–30× (0.0518/0.0056 = 9.25).
- CES survey grid mass: 0.47/0.10 → 0.474/0.102 in Table S3 (E11-2).
- Silhouette printed as 0.719 in both documents.
- "keff = 1.00 exactly" → "1.00 to two decimals (0.997–0.998)"
  (gate_rungs.csv: 0.99716–0.99783).
- CVAP rounding rule stated in full (1–7→4; ≥8→nearest 5; all analyzed
  values ≥1000 are in the nearest-five regime) per the CVAP technical
  documentation PDF (cvaptechdoc URL updated to the www2.census.gov PDF).
- ACS margin-of-error instance narrowed to "consistent with a publication
  grid" (no Census document located that states an explicit MOE rounding
  rule); CVAP remains the documentation-closed attribution.
