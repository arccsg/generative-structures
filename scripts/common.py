"""Shared constants and helpers for the lprofile-geography corpus build."""
import json
import os
import sys

ROOT = "/Users/<ANON>/Projects"
OUT = os.path.join(ROOT, "lprofile-geography")
TABLES = os.path.join(OUT, "tables")
LOGS = os.path.join(OUT, "logs")
CONFIG = os.path.join(OUT, "config")

INCLUDE_EXTS = {".csv", ".tsv", ".txt", ".dat", ".json", ".ndjson", ".jsonl",
                ".parquet", ".xlsx", ".xls", ".zip", ".gz"}

EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "env",
                ".ipynb_checkpoints", "site-packages", ".cache", ".mypy_cache",
                ".pytest_cache", "dist", "build", ".tox"}

# the output tree itself is never walked
EXCLUDE_PATHS = {os.path.join(ROOT, "lprofile-geography")}

HASH_FULL_LIMIT = 200 * 1024 * 1024   # files above this get partial hash
HASH_PARTIAL_BYTES = 64 * 1024 * 1024

DERIVED_NAME_TOKENS = [
    "baseline", "feature", "features", "sweep", "scored", "results", "result",
    "summary", "verification", "prime_stats", "primestats", "equivalence",
    "expb", "msa_", "l_profile", "lprofile", "lprof", "factoriz", "catalog",
    "manifest", "inventory", "z_star", "zstar", "rp_", "report",
]

DERIVED_DIR_NAMES = {"results", "output", "outputs", "figures", "figs",
                     "tables", "derived", "processed", "cache", "checkpoints"}

TABULAR_EXTS = {".csv", ".tsv", ".txt", ".dat", ".parquet", ".xlsx", ".xls"}
STRUCTURED_JSON_EXTS = {".json", ".ndjson", ".jsonl"}

SAMPLE_HEAD_ROWS = 25_000
SAMPLE_TOTAL_ROWS = 50_000


def load_run_config():
    with open(os.path.join(CONFIG, "run_config.json")) as f:
        return json.load(f)


def ext_of(name):
    return os.path.splitext(name)[1].lower()


def top_project(abs_path):
    rel = os.path.relpath(abs_path, ROOT)
    return rel.split(os.sep)[0]
