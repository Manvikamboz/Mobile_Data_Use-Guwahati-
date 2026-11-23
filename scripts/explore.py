from pathlib import Path
import pandas as pd

# Set display options to avoid terminal truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

ROOT = Path(__file__).resolve().parents[1]

train = pd.read_csv(ROOT / "data/raw/train.csv")
later = pd.read_csv(ROOT / "data/raw/later.csv")
holdout = pd.read_csv(ROOT / "data/raw/holdout_features.csv")

print("=== DATASET SHAPES ===")
print(f"train: {train.shape}, later: {later.shape}, holdout: {holdout.shape}\n")

print("=== TRAIN INFO ===")
train.info()
print()

print("=== TRAIN SUMMARY STATISTICS ===")
print(train.describe().T)
print()

print("=== COLUMNS MISSING FROM HOLDOUT (LEAK CANDIDATES) ===")
print(set(train.columns) - set(holdout.columns))
print()

print("=== CATEGORICAL VALUE COUNTS ===")
for c in ["phone_tier", "circle"]:
    print(f"\n--- {c} (TRAIN) ---")
    print(train[c].value_counts(dropna=False))
    print(f"--- {c} (LATER) ---")
    print(later[c].value_counts())

print("\n=== MEDIAN BY CIRCLE ===")
print(train.groupby("circle").median(numeric_only=True).T)
