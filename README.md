# Artifact — L-profile recording-pipeline paper (v10: "When Numerical Structure Reads the Recording Pipeline")

This artifact contains the frozen configurations, decision-rule manifests,
analysis scripts, generated claim tables, figures, and deviation log for
every pre-registered analysis reported in the paper. **Identifying metadata
have been removed for blind review** (see `anonymization_log.txt`); a public
archival DOI will be supplied upon acceptance.

## Directory layout

| Path | Contents |
|---|---|
| `claim_manifest.csv` | **Start here.** One row per manuscript numeric claim → run, script, output table, figure, config, and the SHA-256 of the output that backs it. |
| `manifests/decision_rules.csv` | One row per *pre-registered* decision rule: the rule as frozen, the prediction, the outcome, and a `met`/`missed` verdict. The three reported misfires are rows with verdict `missed`. |
| `deviation_log.csv` | Every logged acquisition/analysis deviation, its impact, and where it is reported in the paper/notes. |
| `configs/` | Every frozen `run_config_*.json`. Predictions and decision rules were written to these files **before** the corresponding computation ran; the confirmatory runs (`gate07`, `hunt09`, `hard10`, `power13`) freeze predictions at script-import time, and the acquisition runs (`emp11`, `whisp12`, `vote14`) freeze them before any network fetch. |
| `scripts/` | The analysis code. `SCRIPTS.md` indexes each script with its role and hash. |
| `tables/` | The generated result tables the manifest points at (builds 07–14). `tables/MANIFEST.sha256` additionally lists **every** table in the full analysis tree (97 files) with hash and size, including intermediate corpus tables too large to bundle. |
| `figures/` | The paper's figures (builds 07–14). `figures/MANIFEST.sha256` lists all 46 tree figures with hash and size. |
| `hashes/SHA256SUMS` | SHA-256 of every file included in this artifact. |
| `anonymization_log.txt` | Exactly what was stripped for blind review. |

## Pre-registration protocol

Each analysis stage wrote its predictions and decision rules into a
`run_config_*.json` **before computation**. A rule states a prediction and
a numeric threshold; the run then reports the outcome and a `met`/`missed`
verdict against the *frozen* rule, with no post-hoc adjustment. Registered
choices that misfired are reported as misfires, not repaired
(`decision_rules.csv`, verdict `missed`): the balanced-pair energy omnibus
(`power13`, P13-2), the inverted atom-count direction (`power13`, P13-3),
and the retired multiplicative-whisper replication (`whisp12`, W12-1).

## Worked example: verify the CVAP attribution result end to end

The paper's headline attribution instance (Abstract; Section 6): a standard
election-forensic test fires at χ² ≈ 12,684 on the modeled CVAP population
series while genuine returns pass, and the encoding coordinates attribute
the alarm to a five-grid.

1. **Find the claim.** In `claim_manifest.csv`, row `C01` (and `C22`):
   `run_id=vote14`, `script=scripts/stage21_vote14.py`,
   `output_table=vote14_beber_scacco.csv`,
   `output_figure=fig14_naive_vs_encoding.png`,
   `config_file=run_config_vote14.json`.
2. **Check the pre-registration.** `configs/run_config_vote14.json` shows
   the Beber–Scacco test was frozen as a *named methodological competitor*
   before any fetch, with the strictly-methodological framing statement.
   `manifests/decision_rules.csv` row `V14-2` records the encoding-vs-modeled
   rule and its `met` verdict.
3. **Read the output.** `tables/vote14_beber_scacco.csv` contains one row per
   series with its Beber–Scacco χ² and p-value: CVAP χ² = 12684 (p ≈ 0);
   every real vote series p ≥ 0.16. `tables/vote14_profiles.csv` gives the
   CVAP encoding signature (rounding mass 0.275 vs enumerated 0.12,
   trailing-grid depth 0.55, mod-25 excess +0.77).
4. **Verify integrity.** The `sha256_of_output` in `claim_manifest.csv` for
   this row equals the hash of `tables/vote14_beber_scacco.csv` (and the
   entry in `tables/MANIFEST.sha256`). Recompute with
   `shasum -a 256 tables/vote14_beber_scacco.csv`.
5. **Re-run (optional).** `scripts/stage21_vote14.py` regenerates the table
   from the public sources named in the acquisition config; deviations
   (e.g. the reported-vs-certified vintage not obtainable keylessly) are in
   `deviation_log.csv`.

The same five steps verify any row of `claim_manifest.csv`.

## Notes

- Absolute paths in scripts/configs were rewritten to `/Users/<ANON>` /
  `<ARTIFACT_ROOT>` for blind review; figures are binary PNGs carrying no
  path metadata and are copied verbatim.
- Two large raw panels (`emp11_panel.csv`, `vote14_panel.csv`, ~4 MB each)
  are bundled; the ~1.9 GB of intermediate corpus tables are not, but every
  one is hashed and sized in `tables/MANIFEST.sha256`.


## v10 additions (JSSAM restructure, 2026-07)

The v10 manuscript restructure added one **exploratory** build and the
main-text figure scripts:

- `scripts/stage22_geo15.py` + `scripts/test_geo15.py` + `configs/run_config_geo15.json` —
  the L1/L3 geography disposition (Blocks A--D: raw geography, controlled
  geography, family-grouped held-out ladders, mechanism stress). The analysis
  plan and disposition rule were frozen in the config **before** results were
  computed; no confirmatory frozen rule covers this build and it is labeled
  exploratory throughout (seed 20260715).
- `tables/geo15_*.csv` — its outputs (geography, controlled stages and
  channels, held-out AUC ladders, stress, per-record structural-zero sample).
- `scripts_v10/` — scripts regenerating main-text Figures 1--6 from the
  bundled tables (see `scripts/SCRIPTS.md`).
- Additional bundled source tables for the figures: `baseline.csv`,
  `derounding_effect.csv`, `deepstrip_effect.csv`, `resonance_controls.csv`,
  `family_profiles_v3.csv`, `residue_structure.csv`.

The frozen v8-era artifact content above (claim manifest, decision rules,
configs, tables, figures, hashes) is unchanged except that
`hashes/SHA256SUMS` now also covers the added files.

## v10.3 additions (review round 1, 2026-07)

Exploratory stages 23--28 (configs `run_config_{loso23,taskpos24,flat25,count26,gate27,powerfp28}.json`,
each frozen before its computation; seeds 20260716--21 are labels, not
timestamps): leave-one-source-out robustness, a task-sensitivity positive
control for the grouped-transfer design, a local-flatness stress simulation,
count-kind spike-in hosts, the fabricated-and-gridded co-occurrence example,
assembled power/false-positive curves, and the ground-truth gate validation
library (`gate27_library.csv`; sensitivity 6/7, specificity 5/6, misses
explained). Outputs under `tables/` with the matching prefixes.

## v10.4 additions and deposit readiness (2026-07)

Round-2 revision support: `scripts/stage29_pairedci.py` +
`configs/run_config_pairedci29.json` (paired family-bootstrap CIs for the
transfer-ladder deltas; output `tables/pairedci29_deltas.csv`), updated
figure scripts (Figure 3 unified to the frozen-arm family-equal estimand),
and `RESULT_PROVENANCE.md` (the number-to-output ledger the supplement
references). This tree is the intended Zenodo snapshot for the JSSAM
submission: anonymization has been applied deposit-wide (see
`anonymization_log.txt`, including the v10.4 entry) and `hashes/SHA256SUMS`
covers every file. `CITATION.cff` carries TODO placeholders that are filled
only on de-anonymization at acceptance.
