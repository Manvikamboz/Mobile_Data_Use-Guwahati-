import pandas as pd
import yaml

try:
    from src.mp238.config import ROOT, load_params
    from src.mp238.data_checks import normalise_text
except ModuleNotFoundError:
    try:
        from mp238.config import ROOT, load_params
        from mp238.data_checks import normalise_text
    except ModuleNotFoundError:
        from config import ROOT, load_params
        from data_checks import normalise_text


def clean_data(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    issues: dict,
    params: dict,
) -> pd.DataFrame:
    """
    Cleans train_df by tackling only the 5 identified data issues in MP238:
      1. Drop leaking column (gb_from_monthly_bill)
      2. Drop duplicate rows (41 copies)
      3. Normalise spelling variants (phone_tier -> flagship, basic, mid)
      4. Correct unit mixup (Rural plan_gb_limit / 1024)
      5. Impute stuck sensor (apps_installed from 2026-04-14 to 2026-04-30)
    """
    df = train_df.copy()
    date_col = params.get("date_col", "date")
    group_col = params.get("group_col", "circle")

    # 1. Leaking column: drop gb_from_monthly_bill
    leak = issues.get("leaking_column", {})
    if leak.get("present"):
        col = leak.get("column", "gb_from_monthly_bill")
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2. Duplicate rows: drop 41 exact duplicate copies
    dups = issues.get("duplicate_rows", {})
    if dups.get("present"):
        df = df.drop_duplicates().reset_index(drop=True)

    # 3. Spelling variants: normalise phone_tier using clean categories from later.csv
    spelling = issues.get("spelling_variants", {})
    if spelling.get("present"):
        col = spelling.get("column", "phone_tier")
        if col in df.columns and col in later_df.columns:
            clean_levels = later_df[col].dropna().unique()
            df[col] = normalise_text(df[col], clean_levels)

    # 4. Unit mixup: divide Rural plan_gb_limit by factor 1024 (MB to GB)
    unit = issues.get("unit_mixup", {})
    if unit.get("present"):
        col = unit.get("column", "plan_gb_limit")
        group_val = unit.get("group_value", "Rural")
        factor = unit.get("factor", 1024)
        if col in df.columns and group_col in df.columns:
            mask = df[group_col] == group_val
            df.loc[mask, col] = df.loc[mask, col] / factor

    # 5. Stuck sensor: impute apps_installed during stuck date range with healthy median
    stuck = issues.get("stuck_sensor", {})
    if stuck.get("present"):
        col = stuck.get("column", "apps_installed")
        start_date = stuck.get("start_date", "2026-04-14")
        end_date = stuck.get("end_date", "2026-04-30")
        if col in df.columns and date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
            stuck_mask = (dates >= pd.to_datetime(start_date)) & (dates <= pd.to_datetime(end_date))
            healthy_median = int(df.loc[~stuck_mask, col].median())
            df.loc[stuck_mask, col] = healthy_median

    return df


def run_cleaning() -> pd.DataFrame:
    """
    Orchestrates the data cleaning pipeline:
    Loads raw datasets and findings.yaml, executes clean_data, and persists
    the cleaned dataset to data/processed/train_clean.csv.
    """
    params = load_params()

    findings_rel = params.get("paths", {}).get("findings", "reports/findings.yaml")
    findings_path = ROOT / findings_rel
    with open(findings_path, "r") as f:
        findings = yaml.safe_load(f)

    train_path = ROOT / params["paths"]["train"]
    later_path = ROOT / params["paths"]["later"]

    train_df = pd.read_csv(train_path)
    later_df = pd.read_csv(later_path)

    issues = findings.get("data_issues", {})
    cleaned_df = clean_data(train_df, later_df, issues, params)

    out_rel = params.get("paths", {}).get("cleaned_train", "data/processed/train_clean.csv")
    out_path = ROOT / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_path, index=False)

    return cleaned_df


if __name__ == "__main__":
    cleaned = run_cleaning()
    params = load_params()
    out_rel = params.get("paths", {}).get("cleaned_train", "data/processed/train_clean.csv")
    out_path = ROOT / out_rel

    print(f"Cleaned dataset saved to: {out_path}")
    print(f"Final shape: {cleaned.shape}")
    print("\nColumns and Non-Null Counts:")
    print(cleaned.info())
    print("\nSample preview:")
    print(cleaned.head(3))
