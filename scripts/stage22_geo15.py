"""Build geo15 (EXPLORATORY): L1/L3 geography disposition analysis.

Supports the v10 manuscript restructure. Re-examines the unresolved L1-vs-L3
Cartesian geography at three observational levels (per-record subsample,
channel, family) in four blocks:

  A  raw geography           -> tables/geo15_geography_channel.csv
  B  controlled geography    -> tables/geo15_controlled.csv
  C  held-out incremental    -> tables/geo15_heldout_auc.csv
  D  mechanism stress        -> tables/geo15_stress.csv
     per-record level check  -> tables/geo15_perrecord_sample.csv

Reuses the frozen v2 corpus features, the frozen digit-conditioned baseline
(tables/baseline.csv, seed 20260702) and the core-magnitude baseline
(tables/baseline_coremag.csv). No frozen confirmatory rule covers this build;
config/run_config_geo15.json freezes the disposition rule and labels the
deltaAUC=0.05 retention threshold as a project-management rule.

Usage: python3 scripts/stage22_geo15.py [null|blocks|heldout|sample|all]
       optional --limit N caps sampled channels for a smoke run.
"""
import json
import math
import random
import zlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import build02_lib as lib

CFG_PATH = os.path.join(common.CONFIG, "run_config_geo15.json")
with open(CFG_PATH) as f:
    CFG = json.load(f)
SEED = CFG["seed"]
T = common.TABLES
LOG = os.path.join(common.LOGS, "stage22_geo15.log")

DIGIT_COLS = [f"digit_frac_d{i}" for i in range(1, 19)]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- profiles
def profile_from_pairs(pairs):
    """(L1, L2, L3, Tail, H2, r_parts) from an exact (prime, exponent) map."""
    ln_n = sum(e * math.log(p) for p, e in pairs)
    c = sorted((e * math.log(p) for p, e in pairs), reverse=True)
    L = [x / ln_n for x in c]
    L1 = L[0]
    L2 = L[1] if len(L) > 1 else 0.0
    L3 = L[2] if len(L) > 2 else 0.0
    return L1, L2, L3, 1.0 - L1 - L2, sum(x * x for x in L), len(L)


def legendre_pairs(n, primes):
    """Exponent map of n! by Legendre's formula."""
    out = []
    for p in primes:
        e, q = 0, p
        while q <= n:
            e += n // q
            q *= p
        if e:
            out.append((p, e))
    return out


def channel_means(values):
    """Channel-level summary of per-record profiles for an int array (>=2).

    Structural-zero treatment: per-record L3 is defined only when the record
    has >= 3 prime-power parts; we report both the all-records mean (zeros
    included, matching the frozen corpus convention) and the conditional mean,
    plus the undefined fraction, so no channel silently mixes the two.
    """
    vals = np.asarray(values, dtype=np.int64)
    vals = vals[vals > 1]
    if len(vals) == 0:
        return None
    uniq, counts = np.unique(vals, return_counts=True)
    tot = counts.sum()
    s = np.zeros(5)
    l3_def_w = 0.0
    l3_def_sum = 0.0
    lt3 = 0.0
    dh = np.zeros(18)
    for v, c in zip(uniq.tolist(), counts.tolist()):
        l1, l2, l3, tail, h2, r = lib.lprofile(v)
        s += np.array([l1, l2, l3, tail, h2]) * c
        if r >= 3:
            l3_def_w += c
            l3_def_sum += l3 * c
        else:
            lt3 += c
        dh[min(len(str(v)), 18) - 1] += c
    m = s / tot
    return dict(n=int(tot), L1=m[0], L2=m[1], L3=m[2], Tail=m[3], H2=m[4],
                L3_cond=(l3_def_sum / l3_def_w) if l3_def_w else np.nan,
                frac_parts_lt3=lt3 / tot, digit_hist=(dh / tot))


def load_baseline():
    b = pd.read_csv(os.path.join(T, "baseline.csv"))
    b["stratum"] = pd.to_numeric(b["stratum"], errors="coerce")
    return b.dropna(subset=["stratum"]).astype({"stratum": int}).set_index("stratum")


def digit_residual(mean_val, digit_frac, base, col):
    """residual = channel mean - sum_d w_d * E[col|d]."""
    exp = float(np.dot(digit_frac, base[col].reindex(range(1, 19)).to_numpy()))
    return mean_val - exp, exp


# ---------------------------------------------------------------- Block A
def _null_channel(task):
    d, k, seed = task
    rng = np.random.default_rng([seed, d, k])
    lo, hi = 10 ** (d - 1), 10 ** d
    vals = rng.integers(max(lo, 2), hi, size=CFG["null_cloud"]["records_per_channel"])
    cm = channel_means(vals)
    return dict(group="uniform_null", channel_id=f"null_d{d}_{k}", domain="null",
                d_mean=float(d), **{k2: cm[k2] for k2 in
                ("n", "L1", "L2", "L3", "Tail", "L3_cond", "frac_parts_lt3")})


def block_a():
    """Raw geography: null cloud, exact constructions, additive controls,
    grid demos, plus slim observational channel rows."""
    log("Block A: null cloud")
    rows = []
    tasks = [(d, k, SEED) for d in CFG["null_cloud"]["strata"]
             for k in range(CFG["null_cloud"]["channels_per_stratum"])]
    with ProcessPoolExecutor(max_workers=6, initializer=lib.init_worker) as ex:
        for r in ex.map(_null_channel, tasks):
            rows.append(r)
    null_df = pd.DataFrame(rows)

    # consistency check vs frozen baseline (machinery validation)
    base = load_baseline()
    tol = CFG["null_cloud"]["consistency_tolerance_vs_frozen_baseline"]
    chk = null_df.groupby(null_df.d_mean.astype(int))[["L1", "L3"]].mean()
    bad = []
    for d, r in chk.iterrows():
        for col, ecol in (("L1", "E_L1"), ("L3", "E_L3")):
            diff = abs(r[col] - base.loc[d, ecol])
            if diff > tol:
                bad.append((d, col, diff))
    if bad:
        log(f"WARNING: null-cloud consistency check failed: {bad}")
    else:
        log(f"null cloud consistent with frozen baseline within {tol}")

    log("Block A: exact constructions + additive controls")
    lib.init_worker()
    random.seed(SEED)
    from sympy import primerange, randprime
    primes_600 = list(primerange(2, 601))
    rng = np.random.default_rng([SEED, 7])

    # factorials and central binomials: closed-form exponent maps
    for name, maker in (
        ("factorial", lambda n: legendre_pairs(n, primes_600)),
        ("central_binomial", lambda n: _binom_pairs(n, primes_600)),
    ):
        profs = [profile_from_pairs(maker(n)) for n in range(60, 601, 20)]
        rows.append(_exact_row(name, profs))

    # balanced products of 2..4 primes, value <= 1e12 (factorizable under stress)
    for k in (2, 3, 4):
        vals = []
        while len(vals) < 120:
            t = 12.0 / k
            ps = [randprime(int(10 ** (t - 0.3)), int(10 ** (t + 0.1))) for _ in range(k)]
            v = 1
            for p in ps:
                v *= p
            if v <= 10 ** 12:
                vals.append(v)
        cm = channel_means(vals)
        rows.append(dict(group=f"exact_product_{k}", channel_id=f"prod{k}",
                         domain="exact", d_mean=12.0,
                         **{k2: cm[k2] for k2 in ("n", "L1", "L2", "L3", "Tail",
                                                  "L3_cond", "frac_parts_lt3")}))
        # recorded versions of the same construction: decimal-grid rounding
        for g in (10, 100, 1000):
            gv = [int(round(v / g)) * g for v in vals]
            cm = channel_means(gv)
            rows.append(dict(group=f"exact_product_{k}_grid{g}",
                             channel_id=f"prod{k}_g{g}", domain="grid_demo",
                             d_mean=12.0,
                             **{k2: cm[k2] for k2 in ("n", "L1", "L2", "L3", "Tail",
                                                      "L3_cond", "frac_parts_lt3")}))

    # additive controls: sums of uniforms and Poisson tallies
    for name, vals in (
        ("additive_sum12", rng.integers(1, 10 ** 4, size=(1500, 12)).sum(axis=1)),
        ("poisson_tally", rng.poisson(50_000, size=1500)),
    ):
        cm = channel_means(vals)
        rows.append(dict(group=name, channel_id=name, domain="additive",
                         d_mean=float(np.mean([len(str(v)) for v in vals])),
                         **{k2: cm[k2] for k2 in ("n", "L1", "L2", "L3", "Tail",
                                                  "L3_cond", "frac_parts_lt3")}))

    out = pd.DataFrame(rows)

    # slim observational rows from the frozen corpus (channel level)
    prof = pd.read_csv(os.path.join(T, "channel_profiles_v3.csv"),
                       usecols=["dataset_id", "dataset_family", "domain",
                                "channel_kind", "L1", "L2", "L3", "Tail",
                                "k_parts_mean", "rounding_mass",
                                "digit_mean_log10", "n_records_used"])
    obs = pd.DataFrame(dict(
        group="observational", channel_id=prof.dataset_id,
        domain=prof.domain, d_mean=prof.digit_mean_log10,
        n=prof.n_records_used, L1=prof.L1, L2=prof.L2, L3=prof.L3,
        Tail=prof.Tail, L3_cond=np.nan, frac_parts_lt3=np.nan,
        dataset_family=prof.dataset_family, channel_kind=prof.channel_kind,
        k_parts_mean=prof.k_parts_mean, rounding_mass=prof.rounding_mass))
    out = pd.concat([out, obs], ignore_index=True)
    out.to_csv(os.path.join(T, "geo15_geography_channel.csv"), index=False)
    log(f"Block A written: {len(out)} rows")


def _binom_pairs(n, primes):
    a = legendre_pairs(2 * n, primes)
    b = {p: e for p, e in legendre_pairs(n, primes)}
    out = []
    for p, e in a:
        r = e - 2 * b.get(p, 0)
        if r > 0:
            out.append((p, r))
    return out


def _exact_row(name, profs):
    a = np.array(profs)
    return dict(group=name, channel_id=name, domain="exact",
                d_mean=float("nan"), n=len(profs),
                L1=a[:, 0].mean(), L2=a[:, 1].mean(), L3=a[:, 2].mean(),
                Tail=a[:, 3].mean(),
                L3_cond=a[a[:, 5] >= 3, 2].mean() if (a[:, 5] >= 3).any() else np.nan,
                frac_parts_lt3=float((a[:, 5] < 3).mean()))


# ---------------------------------------------------------------- Block B

def _restrict_min_families(df, min_fam=5):
    """Keep domains represented by >= min_fam dataset-families (grouped holdout
    needs each class trainable in most folds; singleton-family domains cannot
    transfer by construction and only add noise to the macro average)."""
    fam_per_dom = df.groupby("domain")["dataset_family"].nunique()
    keep = fam_per_dom[fam_per_dom >= min_fam].index
    return df[df.domain.isin(keep)].copy()

def _stage_auc(X, y, groups, seed, n_boot=200):
    """OOF macro-OVR AUC of domain from the stage coordinates alone,
    grouped by dataset_family; family-grouped bootstrap CI."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(X).all(axis=1)
    X, y, groups = X[ok], y[ok], groups[ok]
    classes = np.unique(y)
    proba = np.full((len(y), len(classes)), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(random_state=seed, early_stopping=False)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        cols = {c: i for i, c in enumerate(clf.classes_)}
        for j, c in enumerate(classes):
            proba[te, j] = p[:, cols[c]] if c in cols else 0.0

    def macro_auc(idx):
        aucs = []
        for j, c in enumerate(classes):
            yy = (y[idx] == c).astype(int)
            if 0 < yy.sum() < len(yy):
                aucs.append(roc_auc_score(yy, proba[idx, j]))
        return float(np.mean(aucs)) if aucs else np.nan

    point = macro_auc(np.arange(len(y)))
    fams = pd.unique(groups)
    fam_idx = {f: np.where(groups == f)[0] for f in fams}
    rng = np.random.default_rng([SEED, 99])
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(fams, size=len(fams), replace=True)
        idx = np.concatenate([fam_idx[f] for f in pick])
        boots.append(macro_auc(idx))
    boots = np.array([b for b in boots if np.isfinite(b)])
    return point, float(np.percentile(boots, 5)), float(np.percentile(boots, 95)), int(ok.sum())


def block_b():
    """Controlled geography at channel level, full frozen corpus."""
    log("Block B: controlled geography")
    prof = pd.read_csv(os.path.join(T, "channel_profiles_v3.csv"))
    res = pd.read_csv(os.path.join(T, "residue_structure.csv"),
                      usecols=["dataset_id", "resonance_score", "lattice_gcd"])
    df = prof.merge(res, on="dataset_id", how="left")
    lab = pd.read_csv(CFG["inputs"]["labels"]).rename(columns={"channel_id": "dataset_id"})
    df = df.merge(lab[["dataset_id", "corpus_role"]], on="dataset_id", how="left")
    df = df[(df.domain != "unresolved") & (df.corpus_role != "synthetic_control")]
    df = _restrict_min_families(df)

    base = load_baseline()
    W = df[DIGIT_COLS].to_numpy()
    W = W / W.sum(axis=1, keepdims=True)
    for col, ecol in (("L1", "E_L1"), ("L3", "E_L3"), ("Tail", "E_Tail")):
        exp = W @ base[ecol].reindex(range(1, 19)).to_numpy()
        df[f"g15_d{col}"] = df[col] - exp
    # sanity: our dL1 should match the stored dL1
    dd = (df["g15_dL1"] - df["dL1"]).abs()
    log(f"dL1 recomputation max abs diff vs stored: {dd.max():.5f} "
        f"(median {dd.median():.6f})")

    # S4: residualize de-rounded coords on residue-quantization variables
    from sklearn.linear_model import LinearRegression
    s4 = df[["dL1_der", "dTail_der", "resonance_score", "lattice_gcd"]].copy()
    s4["log_gcd"] = np.log10(s4.lattice_gcd.clip(lower=1))
    m_ok = s4[["dL1_der", "dTail_der", "resonance_score", "log_gcd"]].notna().all(axis=1)
    Z = s4.loc[m_ok, ["resonance_score", "log_gcd"]].to_numpy()
    for c in ("dL1_der", "dTail_der"):
        lr = LinearRegression().fit(Z, s4.loc[m_ok, c])
        df.loc[m_ok, f"g15_{c}_resid"] = s4.loc[m_ok, c] - lr.predict(Z)

    stages = [
        ("S0_raw", ["L1", "L3"]),
        ("S1_digit", ["g15_dL1", "g15_dL3"]),
        ("S2_deround", ["dL1_der", "dTail_der"]),
        ("S4_residue", ["g15_dL1_der_resid", "g15_dTail_der_resid"]),
    ]
    y = df.domain.to_numpy()
    groups = df.dataset_family.to_numpy()
    rows = []
    from sklearn.metrics import silhouette_score
    for name, cols in stages:
        X = df[cols].to_numpy(dtype=float)
        auc, lo, hi, n_used = _stage_auc(X, y, groups, SEED)
        # family-level silhouette by domain (families as points, >=3-family domains)
        fam = df.assign(x0=X[:, 0], x1=X[:, 1]).groupby(
            ["dataset_family", "domain"], as_index=False)[["x0", "x1"]].mean()
        fam = fam.dropna()
        keep = fam.domain.map(fam.domain.value_counts()) >= 3
        fam = fam[keep]
        Xf = (fam[["x0", "x1"]] - fam[["x0", "x1"]].mean()) / fam[["x0", "x1"]].std()
        sil = float(silhouette_score(Xf, fam.domain)) if fam.domain.nunique() > 1 else np.nan
        rows.append(dict(stage=name, coords="+".join(cols), macro_auc=auc,
                         auc_lo=lo, auc_hi=hi, n_channels=n_used,
                         silhouette_family=sil, n_families=len(fam)))
        log(f"  {name}: macro-AUC {auc:.3f} [{lo:.3f},{hi:.3f}], "
            f"family silhouette {sil:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(T, "geo15_controlled.csv"), index=False)
    # persist the per-channel controlled coordinates for the figure scripts
    keep_cols = ["dataset_id", "dataset_family", "domain", "channel_kind",
                 "L1", "L3", "Tail", "k_parts_mean", "rounding_mass",
                 "g15_dL1", "g15_dL3", "g15_dTail", "dL1_der", "dTail_der",
                 "g15_dL1_der_resid", "g15_dTail_der_resid",
                 "resonance_score", "lattice_gcd", "digit_mean_log10"]
    df[keep_cols].to_csv(os.path.join(T, "geo15_controlled_channels.csv"), index=False)
    log("Block B written")


# ---------------------------------------------------------------- Block C
def block_c():
    """Held-out incremental information of L1/L3 beyond recording variables."""
    log("Block C: held-out ladders")
    feat = pd.read_csv(CFG["inputs"]["features"])
    lab = pd.read_csv(CFG["inputs"]["labels"])
    enc = pd.read_csv(CFG["inputs"]["encoding"])[
        ["channel_id", "decimals", "tick_log10"]]
    res = pd.read_csv(os.path.join(T, "residue_structure.csv"),
                      usecols=["dataset_id", "resonance_score", "lattice_gcd"]
                      ).rename(columns={"dataset_id": "channel_id"})
    df = (feat.merge(lab.drop(columns=["dataset_family"]), on="channel_id")
              .merge(enc, on="channel_id", how="left")
              .merge(res, on="channel_id", how="left"))
    df = df[(df.domain != "unresolved") & (df.corpus_role != "synthetic_control")]
    df = _restrict_min_families(df)
    df["log10_lattice_gcd"] = np.log10(df.lattice_gcd.clip(lower=1))

    R = ["mean_log10_digits", "sd_log10_digits", "rounding_mass", "v3_share",
         "decimals", "tick_log10", "resonance_score", "log10_lattice_gcd"]
    ladders = [
        ("R_recording", R),
        ("R_plus_L1", R + ["L1"]),
        ("R_plus_L1_L3", R + ["L1", "L3"]),
        ("R_full_profile", R + ["L1", "L2", "L3", "Tail", "H2", "keff"]),
    ]
    rows = []
    for target, kind in (("domain", "multiclass"), ("count_vs_amount", "binary")):
        if target == "domain":
            sub = df
            y = sub.domain.to_numpy()
        else:
            sub = df[df.channel_kind.isin(["count", "amount"])]
            y = (sub.channel_kind == "amount").astype(int).to_numpy()
        groups = sub.dataset_family.to_numpy()
        for name, cols in ladders:
            X = sub[cols].to_numpy(dtype=float)
            X = np.nan_to_num(X, nan=-1.0)
            auc, lo, hi, n_used = _stage_auc(X, y, groups, SEED, n_boot=200)
            rows.append(dict(task=target, ladder=name, n_features=len(cols),
                             macro_auc=auc, auc_lo=lo, auc_hi=hi, n=n_used))
            log(f"  {target} / {name}: AUC {auc:.4f} [{lo:.4f},{hi:.4f}]")
    out = pd.DataFrame(rows)
    for task in out.task.unique():
        m = out.task == task
        ref = out.loc[m & (out.ladder == "R_recording"), "macro_auc"].iloc[0]
        out.loc[m, "delta_vs_R"] = out.loc[m, "macro_auc"] - ref
        refl1 = out.loc[m & (out.ladder == "R_plus_L1"), "macro_auc"].iloc[0]
        out.loc[m, "delta_vs_R_L1"] = out.loc[m, "macro_auc"] - refl1
    out.to_csv(os.path.join(T, "geo15_heldout_auc.csv"), index=False)
    log("Block C written")


# ---------------------------------------------------------------- Block D
def strip_pairs(pairs, cut):
    return [(p, e) for p, e in pairs if p > cut]


def deround_int(n):
    while n % 2 == 0:
        n //= 2
    while n % 5 == 0:
        n //= 5
    return n


def _transform(vals, kind, g0):
    v = np.asarray(vals, dtype=np.int64)
    if kind == "identity":
        return v
    if kind == "plus1":
        return v + 1
    if kind == "round10":
        return (np.round(v / 10).astype(np.int64)) * 10
    if kind == "round100":
        return (np.round(v / 100).astype(np.int64)) * 100
    if kind == "grid_quotient":
        g = max(int(g0), 1)
        return v[v % g == 0] // g
    raise ValueError(kind)


def _sample_channel(task):
    """Per-channel: read source, per-record profiles under each transform."""
    rec, seed, per_rec_dump = task
    try:
        df, sampled = lib.read_table(rec["file_path"], rec["archive_member"],
                                     rec["sheet_or_table"], rec["ext"], seed)
        if df is None or rec["column_name"] not in df.columns:
            return None
        vals, _ = lib.coerce_ints(df[rec["column_name"]], rec["monetary"])
    except Exception as e:
        return ("err", rec["dataset_id"], repr(e)[:120])
    if len(vals) < 100:
        return None
    rng = np.random.default_rng([seed, zlib.crc32(rec["dataset_id"].encode())])
    if len(vals) > rec["cap"]:
        vals = rng.choice(vals, size=rec["cap"], replace=False)

    out_rows = []
    per_rec = []
    for kind in ("identity", "plus1", "round10", "round100", "grid_quotient"):
        tv = _transform(vals, kind, rec["g0"])
        tv = tv[tv > 1]
        if len(tv) < 50:
            continue
        cm = channel_means(tv)
        cm.pop("digit_hist")
        out_rows.append(dict(dataset_id=rec["dataset_id"], transform=kind,
                             domain=rec["domain"], family=rec["dataset_family"],
                             channel_kind=rec["channel_kind"], **cm))
    # derived transforms reuse identity's factorizations via lib memo
    uniq, counts = np.unique(vals[vals > 1], return_counts=True)
    for kind, op in (("deround", None), ("strip7", None)):
        s = np.zeros(5)
        lt3 = 0.0
        tot = 0
        core_dh = np.zeros(30)
        for v, c in zip(uniq.tolist(), counts.tolist()):
            pairs = lib.factor_pairs(int(v))
            cut = 5 if kind == "deround" else 7
            kp = [(p, e) for p, e in pairs if p > cut] if kind == "strip7" \
                else [(p, e) for p, e in pairs if p not in (2, 5)]
            if not kp:
                continue
            l1, l2, l3, tail, h2, r = profile_from_pairs(kp)
            s += np.array([l1, l2, l3, tail, h2]) * c
            lt3 += c if r < 3 else 0
            tot += c
            kd = int(sum(e * math.log10(p) for p, e in kp)) + 1
            core_dh[min(kd, 29)] += c
        if tot < 50:
            continue
        m = s / tot
        out_rows.append(dict(dataset_id=rec["dataset_id"], transform=kind,
                             domain=rec["domain"], family=rec["dataset_family"],
                             channel_kind=rec["channel_kind"], n=int(tot),
                             L1=m[0], L2=m[1], L3=m[2], Tail=m[3], H2=m[4],
                             L3_cond=np.nan, frac_parts_lt3=lt3 / tot,
                             core_digit_mean=float(
                                 np.average(np.arange(30), weights=core_dh))))
    if per_rec_dump:
        sub = rng.choice(uniq, size=min(300, len(uniq)), replace=False,
                         p=counts / counts.sum())
        for v in sub.tolist():
            l1, l2, l3, tail, h2, r = lib.lprofile(int(v))
            per_rec.append(dict(dataset_id=rec["dataset_id"],
                                domain=rec["domain"], value_digits=len(str(v)),
                                L1=l1, L3=l3, r_parts=r))
    return ("ok", out_rows, per_rec)


def block_d(limit=None):
    """Per-record subsample + mechanism stress (Block D and level check)."""
    log("Block D: sampling channels")
    corpus = pd.read_csv(CFG["inputs"]["corpus"])
    res = pd.read_csv(os.path.join(T, "residue_structure.csv"),
                      usecols=["dataset_id", "lattice_gcd"])
    corpus = corpus.merge(res, on="dataset_id", how="left")
    corpus = corpus[corpus.domain != "unresolved"]
    avail = corpus[corpus.file_path.map(lambda p: os.path.exists(p))].copy()
    log(f"available on disk: {len(avail)} / {len(corpus)} channels")

    ps = CFG["per_record_sample"]
    rng = np.random.default_rng([SEED, 3])
    picks = []
    for dom, dgrp in avail.groupby("domain"):
        fams = dgrp.dataset_family.unique()
        rng.shuffle(fams)
        for fam in fams[: ps["max_families_per_domain"]]:
            fg = dgrp[dgrp.dataset_family == fam]
            take = fg.sample(n=min(ps["max_channels_per_family"], len(fg)),
                             random_state=int(rng.integers(2 ** 31)))
            picks.append(take)
    sample = pd.concat(picks, ignore_index=True)
    if limit:
        sample = sample.head(limit)
    log(f"sampled {len(sample)} channels across {sample.domain.nunique()} domains")

    tasks = []
    dump_ids = set(sample.groupby("domain").head(3).dataset_id)
    for _, r in sample.iterrows():
        ext = os.path.splitext(r.archive_member if isinstance(r.archive_member, str)
                               and r.archive_member else r.file_path)[1].lower()
        monetary = bool(r.looks_monetary) or (r.channel_kind == "amount")
        tasks.append((dict(dataset_id=r.dataset_id, file_path=r.file_path,
                           archive_member=r.archive_member if isinstance(r.archive_member, str) else None,
                           sheet_or_table=r.sheet_or_table if isinstance(r.sheet_or_table, str) else None,
                           ext=ext, column_name=r.column_name, monetary=monetary,
                           domain=r.domain, dataset_family=r.dataset_family,
                           channel_kind=r.channel_kind,
                           g0=r.lattice_gcd if np.isfinite(r.lattice_gcd) else 1,
                           cap=ps["max_records_per_channel"]),
                      SEED, r.dataset_id in dump_ids))

    stress_rows, per_rec_rows, errs = [], [], []
    with ProcessPoolExecutor(max_workers=6, initializer=lib.init_worker) as ex:
        for i, out in enumerate(ex.map(_sample_channel, tasks)):
            if out is None:
                continue
            if out[0] == "err":
                errs.append(out)
                continue
            stress_rows.extend(out[1])
            per_rec_rows.extend(out[2])
            if (i + 1) % 50 == 0:
                log(f"  processed {i + 1}/{len(tasks)} channels")
    log(f"read errors: {len(errs)}")

    # exact constructions under the same stress (destruction demonstration)
    lib.init_worker()
    random.seed(SEED + 1)
    from sympy import randprime
    rng2 = np.random.default_rng([SEED, 11])
    vals = []
    while len(vals) < 120:
        a = randprime(10 ** 5, 10 ** 6)
        b = randprime(10 ** 5, 10 ** 6)
        vals.append(a * b)
    vals3 = []
    while len(vals3) < 120:
        v = randprime(10 ** 4, 10 ** 5) * randprime(10 ** 4, 10 ** 5) * randprime(10 ** 3, 10 ** 4)
        vals3.append(v)
    for name, vs in (("exact_product_2", vals), ("exact_product_3", vals3)):
        for kind in ("identity", "plus1", "round10", "round100"):
            tv = _transform(np.array(vs, dtype=np.int64), kind, 1)
            cm = channel_means(tv)
            cm.pop("digit_hist")
            stress_rows.append(dict(dataset_id=name, transform=kind,
                                    domain="exact", family=name,
                                    channel_kind="exact", **cm))

    sdf = pd.DataFrame(stress_rows)

    sdf.to_csv(os.path.join(T, "geo15_stress.csv"), index=False)
    pd.DataFrame(per_rec_rows).to_csv(
        os.path.join(T, "geo15_perrecord_sample.csv"), index=False)
    log(f"Block D written: {len(sdf)} stress rows, {len(per_rec_rows)} per-record rows")

    # separability under transforms (subsample level, grouped by family)
    rows = []
    for kind in sdf["transform"].unique():
        sub = sdf[(sdf["transform"] == kind) & (sdf.domain != "exact")].dropna(subset=["L1", "L3"])
        if sub.domain.nunique() < 3 or len(sub) < 60:
            continue
        auc, lo, hi, n_used = _stage_auc(sub[["L1", "L3"]].to_numpy(float),
                                         sub.domain.to_numpy(),
                                         sub.family.to_numpy(), SEED, n_boot=100)
        rows.append(dict(transform=kind, macro_auc=auc, auc_lo=lo, auc_hi=hi,
                         n_channels=n_used))
        log(f"  transform {kind}: (L1,L3) domain macro-AUC {auc:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(T, "geo15_stress_separability.csv"),
                              index=False)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit"):
            limit = int(a.split("=")[1])
    which = args[0] if args else "all"
    log(f"=== geo15 start: {which} (seed {SEED}, exploratory) ===")
    if which in ("null", "all", "a"):
        block_a()
    if which in ("blocks", "all", "b"):
        block_b()
    if which in ("heldout", "all", "c"):
        block_c()
    if which in ("sample", "all", "d"):
        block_d(limit)
    log("=== geo15 done ===")
