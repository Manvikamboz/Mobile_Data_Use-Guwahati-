import numpy as np
import pandas as pd
import pytest

from src.mp238.data_checks import (
    normalise_text,
    check_leaking_column,
    check_unit_mixup,
    check_duplicates,
    check_stuck_sensor,
    check_sentinel,
    check_spelling_variants,
)


def test_normalise_text():
    """Verify normalise_text handles abbreviations, fuzzy spelling, and casing."""
    clean_levels = ["flagship", "basic", "mid"]
    series = pd.Series(["fla.", "FLAGSHIP", "basic ", "m-id", "unknown_tier"])
    fixed = normalise_text(series, clean_levels)

    assert fixed[0] == "flagship"  # prefix match "fla." -> "flagship"
    assert fixed[1] == "flagship"  # uppercase -> "flagship"
    assert fixed[2] == "basic"     # trailing space -> "basic"
    assert fixed[3] == "mid"       # punctuation "m-id" -> "mid"


def test_check_leaking_column():
    """Verify check_leaking_column detects columns missing from holdout with high corr."""
    target_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    train_df = pd.DataFrame({
        "feature_ok": [5.0] * 10,
        "leaking_feature": [t * 1.01 for t in target_values],  # corr ~ 1.0 with target
        "gb_per_day": target_values,
    })
    holdout_df = pd.DataFrame({
        "feature_ok": [5.0] * 5,
    })

    result = check_leaking_column(
        train=train_df,
        holdout=holdout_df,
        target="gb_per_day",
        threshold=0.95,
    )
    assert result["present"] is True
    assert result["column"] == "leaking_feature"


def test_check_unit_mixup():
    """Verify check_unit_mixup detects unit factor discrepancies (e.g. 1024 MB to GB)."""
    # 3 circles matching the real dataset (Urban, Semi-urban, Rural)
    df = pd.DataFrame({
        "circle": ["Urban"] * 20 + ["Semi-urban"] * 20 + ["Rural"] * 20,
        "plan_gb_limit": [2.0] * 20 + [2.0] * 20 + [2048.0] * 20,  # 1024x factor in Rural
        "wifi_hours": [3.0] * 60,
    })

    result = check_unit_mixup(
        df=df,
        group_col="circle",
        num_cols=["plan_gb_limit", "wifi_hours"],
    )
    assert result["present"] is True
    assert result["column"] == "plan_gb_limit"
    assert result["group_value"] == "Rural"
    assert result["factor"] == 1024


def test_check_duplicates():
    """Verify check_duplicates counts duplicate rows accurately."""
    df_no_dups = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert check_duplicates(df_no_dups)["present"] is False

    df_dups = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    res = check_duplicates(df_dups)
    assert res["present"] is True
    assert res["count"] == 1


def test_check_stuck_sensor():
    """Verify check_stuck_sensor finds constant sensor value runs >= min_run."""
    dates = pd.date_range("2026-01-01", periods=30)
    # Value 99 stuck for 25 consecutive days (min_run=20)
    values = [10.0] * 5 + [99.0] * 25
    df = pd.DataFrame({
        "date": dates,
        "apps_installed": values,
    })

    result = check_stuck_sensor(
        df=df,
        num_cols=["apps_installed"],
        date_col="date",
        min_run=20,
    )
    assert result["present"] is True
    assert result["column"] == "apps_installed"
    assert result["start_date"] == "2026-01-06"
    assert result["end_date"] == "2026-01-30"


def test_check_sentinel():
    """Verify check_sentinel detects sentinel values like -99 or 999."""
    df_clean = pd.DataFrame({"age": [20, 25, 30, 35]})
    assert check_sentinel(df_clean, ["age"])["present"] is False

    df_sentinel = pd.DataFrame({"age": [20, -99, -99, -99, 30]})
    res = check_sentinel(df_sentinel, ["age"], min_count=3)
    assert res["present"] is True
    assert res["column"] == "age"
    assert res["value"] == -99
    assert res["count"] == 3


def test_check_spelling_variants():
    """Verify check_spelling_variants identifies unknown category levels."""
    train_df = pd.DataFrame({"phone_tier": ["basic", "mid", "flagship", "fla."]})
    later_df = pd.DataFrame({"phone_tier": ["basic", "mid", "flagship"]})

    res = check_spelling_variants(train_df, later_df, ["phone_tier"])
    assert res["present"] is True
    assert res["column"] == "phone_tier"
    assert res["true_levels"] == 3
