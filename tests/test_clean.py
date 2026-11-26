import pandas as pd
import pytest

from src.mp238.clean import clean_data


@pytest.fixture
def clean_test_setup():
    """Build small synthetic in-memory data for tests (no external CSVs required)."""
    params = {
        "date_col": "date",
        "group_col": "circle",
    }

    issues = {
        "leaking_column": {
            "present": True,
            "column": "gb_from_monthly_bill",
        },
        "duplicate_rows": {
            "present": True,
            "count": 1,
        },
        "spelling_variants": {
            "present": True,
            "column": "phone_tier",
            "true_levels": 3,
        },
        "unit_mixup": {
            "present": True,
            "column": "plan_gb_limit",
            "group_value": "Rural",
            "factor": 1024,
        },
        "stuck_sensor": {
            "present": True,
            "column": "apps_installed",
            "start_date": "2026-04-14",
            "end_date": "2026-04-16",
        },
    }

    train_data = {
        "record_id": ["R001", "R002", "R003", "R004", "R004"],  # Contains 1 duplicate (R004)
        "date": ["2026-04-10", "2026-04-14", "2026-04-15", "2026-04-20", "2026-04-20"],
        "age": [25, 30, 35, 40, 40],
        "screen_hours": [4.0, 5.0, 6.0, 7.0, 7.0],
        "apps_installed": [20, 999, 999, 40, 40],  # Stuck sensor during 2026-04-14 to 2026-04-16
        "plan_gb_limit": [2.0, 2048.0, 1.5, 3.0, 3.0],  # Rural row has 2048 MB instead of 2 GB
        "video_minutes": [100.0, 150.0, 200.0, 250.0, 250.0],
        "wifi_hours": [2.0, 3.0, 1.0, 4.0, 4.0],
        "phone_tier": ["mid", "fla.", "basic", "MID", "MID"],  # Misspellings: "fla.", "MID"
        "circle": ["Urban", "Rural", "Semi-urban", "Urban", "Urban"],
        "gb_from_monthly_bill": [2.1, 3.4, 1.8, 4.0, 4.0],  # Leaking column
        "gb_per_day": [2.0, 3.2, 1.7, 3.8, 3.8],
    }
    train_df = pd.DataFrame(train_data)

    later_data = {
        "phone_tier": ["flagship", "basic", "mid"],
        "circle": ["Urban", "Semi-urban", "Rural"],
    }
    later_df = pd.DataFrame(later_data)

    return train_df, later_df, issues, params


def test_clean_data_drops_leaking_column(clean_test_setup):
    """Verify clean_data drops leaking column gb_from_monthly_bill."""
    train_df, later_df, issues, params = clean_test_setup
    assert "gb_from_monthly_bill" in train_df.columns

    cleaned = clean_data(train_df, later_df, issues, params)
    assert "gb_from_monthly_bill" not in cleaned.columns


def test_clean_data_removes_duplicates(clean_test_setup):
    """Verify clean_data drops exact duplicate rows."""
    train_df, later_df, issues, params = clean_test_setup
    assert len(train_df) == 5

    cleaned = clean_data(train_df, later_df, issues, params)
    assert len(cleaned) == 4
    assert cleaned["record_id"].tolist() == ["R001", "R002", "R003", "R004"]


def test_clean_data_normalises_spelling(clean_test_setup):
    """Verify phone_tier variants ('fla.', 'MID') are mapped to standard levels."""
    train_df, later_df, issues, params = clean_test_setup
    cleaned = clean_data(train_df, later_df, issues, params)

    valid_tiers = set(later_df["phone_tier"].unique())
    for val in cleaned["phone_tier"]:
        assert val in valid_tiers
    # "fla." should map to "flagship", "MID" to "mid"
    assert cleaned.loc[cleaned["record_id"] == "R002", "phone_tier"].iloc[0] == "flagship"
    assert cleaned.loc[cleaned["record_id"] == "R004", "phone_tier"].iloc[0] == "mid"


def test_clean_data_fixes_unit_mixup(clean_test_setup):
    """Verify Rural plan_gb_limit in MB is divided by 1024 to convert to GB."""
    train_df, later_df, issues, params = clean_test_setup
    cleaned = clean_data(train_df, later_df, issues, params)

    rural_plan = cleaned.loc[cleaned["circle"] == "Rural", "plan_gb_limit"].iloc[0]
    assert rural_plan == pytest.approx(2.0, abs=1e-3)


def test_clean_data_imputes_stuck_sensor(clean_test_setup):
    """Verify stuck sensor during target date range is imputed with healthy median."""
    train_df, later_df, issues, params = clean_test_setup
    cleaned = clean_data(train_df, later_df, issues, params)

    # In healthy rows (R001: 20, R004: 40), median is 30
    stuck_val = cleaned.loc[cleaned["record_id"] == "R002", "apps_installed"].iloc[0]
    assert stuck_val == 30
