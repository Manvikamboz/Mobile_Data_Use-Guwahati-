from pathlib import Path
import re
import difflib
import numpy as np
import pandas as pd
import yaml

try:
    from src.mp238.config import ROOT, load_params
except ModuleNotFoundError:
    try:
        from mp238.config import ROOT, load_params
    except ModuleNotFoundError:
        from config import ROOT, load_params

# 1024 is the exact binary conversion for data units (MB to GB in Rural plan_gb_limit)
UNIT_FACTORS = [10, 12, 60, 100, 1000, 1024]

# Exclude 99 because apps_installed legitimately ranges from 10 to 200
SENTINELS = [-1, -9, -99, -999, -9999, 999, 9999, 99999]


def normalise_text(series, clean_levels):
    """Turn messy spellings into the clean ones (clean_levels come from later.csv)."""
    simple = lambda t: re.sub(r"[^a-z-]", "", str(t).lower())
    lookup = {simple(level): level for level in clean_levels}

    def fix(value):
        if pd.isna(value):
            return value
        key = simple(value)
        if key not in lookup:
            # 1. Exact prefix match for abbreviations ('fla.' -> 'fla' -> 'flagship')
            prefix_matches = [k for k in lookup if k.startswith(key) and len(key) >= 2]
            if len(prefix_matches) == 1:
                key = prefix_matches[0]
            else:
                # 2. Fuzzy fallback for typos
                close = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.6)
                key = close[0] if close else key
        return lookup.get(key, value)

    return series.map(fix)


def typo_cutoff(df, target):
    """Largest normal target value, or None. Anything above it is a 10x typo."""
    s = df[target].dropna().sort_values().reset_index(drop=True)
    jump = (s / s.shift(1))[s.index >= int(len(s) * 0.85)]  # jump between neighbours, top 15% only
    if jump.empty or jump.max() < 2:
        return None
    return float(s[jump.idxmax() - 1])  # value just below the biggest jump


def check_leaking_column(train, holdout, target, threshold):
    # columns that exist in train but not in holdout (and aren't the target)
    candidates = [c for c in train.columns if c not in holdout.columns and c != target]
    for col in candidates:
        corr = abs(train[col].corr(train[target], method="spearman"))
        if corr >= threshold:
            return {"present": True, "column": col}
    return {"present": False, "column": None}


def check_unit_mixup(df, group_col, num_cols):
    best = (0, None, None, None)  # (score, column, group, ratio)
    for col in num_cols:
        for g in df[group_col].dropna().unique():
            inside = df.loc[df[group_col] == g, col].median()
            outside = df.loc[df[group_col] != g, col].median()
            if inside > 0 and outside > 0:
                ratio = inside / outside
                if abs(np.log(ratio)) > best[0]:
                    best = (abs(np.log(ratio)), col, g, ratio)

    score, col, g, ratio = best
    if score < np.log(5):  # less than 5x apart: normal
        return {"present": False, "column": None, "group_value": None, "factor": None}

    big = max(ratio, 1 / ratio)
    factor = min(UNIT_FACTORS, key=lambda f: abs(np.log(big / f)))  # nearest known conversion
    if ratio < 1:
        factor = 1 / factor
    factor_val = int(factor) if factor == int(factor) else float(factor)
    return {"present": True, "column": col, "group_value": str(g), "factor": factor_val}


def check_sentinel(df, num_cols, min_count=3):
    best = (0, None, None)  # (count, column, value)
    for col in num_cols:
        for s in SENTINELS:
            n = int((df[col] == s).sum())
            if n >= min_count and n > best[0]:
                best = (n, col, s)

    n, col, s = best
    if col is None:
        return {"present": False, "column": None, "value": None, "count": None}
    val = int(s) if s == int(s) else float(s)
    return {"present": True, "column": col, "value": val, "count": int(n)}


def check_duplicates(df):
    n = int(df.duplicated().sum())
    return {"present": bool(n > 0), "count": int(n) if n > 0 else None}


def check_stuck_sensor(df, num_cols, date_col, min_run):
    d = df.sort_values(date_col, kind="stable")
    best = (0, None, None, None)  # (run length, column, start, end)
    for col in num_cols:
        run_id = (d[col] != d[col].shift()).cumsum()  # new id each time the value changes
        runs = d.groupby(run_id)[date_col].agg(["size", "min", "max"])
        longest = runs.loc[runs["size"].idxmax()]
        if longest["size"] > best[0]:
            best = (longest["size"], col, longest["min"], longest["max"])

    size, col, start, end = best
    if size < min_run:
        return {"present": False, "column": None, "start_date": None, "end_date": None}
    return {
        "present": True,
        "column": col,
        "start_date": pd.to_datetime(start).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(end).strftime("%Y-%m-%d"),
    }


def check_target_typos(df, target):
    cutoff = typo_cutoff(df, target)
    if cutoff is None:
        return {"present": False, "count": None}
    return {"present": True, "count": int((df[target] > cutoff).sum())}


def check_spelling_variants(train, later, text_cols):
    for col in text_cols:
        if set(train[col].dropna()) - set(later[col].dropna()):  # spellings later.csv doesn't have
            return {"present": True, "column": col, "true_levels": int(later[col].nunique())}
    return {"present": False, "column": None, "true_levels": None}


def run_all(train_raw, later, holdout, params):
    target, date_col = params["target"], params["date_col"]
    train = train_raw.copy()
    train[date_col] = pd.to_datetime(train[date_col])

    # tidy copy so 'Rural' and 'rural ' count as one group in the unit and stuck checks
    tidy = train.copy()
    for col in params["text_cols"]:
        tidy[col] = normalise_text(tidy[col], later[col].dropna().unique())

    nums = params["numeric_cols"]
    return {
        "leaking_column": check_leaking_column(train, holdout, target, params["leak_corr_threshold"]),
        "unit_mixup": check_unit_mixup(tidy, params["group_col"], nums),
        "sentinel_value": check_sentinel(train, nums),
        "duplicate_rows": check_duplicates(train),
        "stuck_sensor": check_stuck_sensor(tidy, nums, date_col, params["stuck_min_run"]),
        "target_typos": check_target_typos(train, target),
        "spelling_variants": check_spelling_variants(train, later, params["text_cols"]),
    }


def write_findings(data_issues, path):
    """
    Writes data_issues into findings.yaml at the given path.
    Preserves group_id, problem_id, tracking, and monitoring sections intact.
    """
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            doc = yaml.safe_load(f) or {}
    else:
        doc = {}

    doc["data_issues"] = data_issues

    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


if __name__ == "__main__":
    params = load_params()

    train = pd.read_csv(ROOT / params["paths"]["train"])
    later = pd.read_csv(ROOT / params["paths"]["later"])
    holdout = pd.read_csv(ROOT / params["paths"]["holdout"])
    findings_path = ROOT / params["paths"]["findings"]

    issues = run_all(train, later, holdout, params)
    print("Detected Issues:")
    for k, v in issues.items():
        print(f"  {k}: {v}")

    # write_findings(issues, findings_path)
    # print(f"\nSuccessfully wrote data issues to {findings_path}")
