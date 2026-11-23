import hashlib
from pathlib import Path
import joblib
import pandas as pd
import yaml

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

try:
    from src.mp238.config import ROOT, load_params
except ModuleNotFoundError:
    try:
        from mp238.config import ROOT, load_params
    except ModuleNotFoundError:
        from config import ROOT, load_params


def compute_sha256(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of the raw train file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_candidates(params: dict) -> list[tuple[str, str, object, dict]]:

    training_cfg = params["training"]
    rs = training_cfg.get("random_state", 42)
    models_cfg = training_cfg["models"]

    candidates = []

    # 1. Dummy baseline
    dummy_strategy = models_cfg["dummy"]["strategy"]
    candidates.append((
        "dummy_baseline",
        "dummy",
        DummyRegressor(strategy=dummy_strategy),
        models_cfg["dummy"],
    ))

    # 2. Linear Regression
    candidates.append((
        "linear_regression",
        "linear_regression",
        LinearRegression(),
        {},
    ))

    # 3. Decision Trees
    dt_cfg = models_cfg["decision_tree"]
    for depth in dt_cfg["max_depth"]:
        run_name = f"decision_tree_d{depth if depth is not None else 'None'}"
        candidates.append((
            run_name,
            "decision_tree",
            DecisionTreeRegressor(max_depth=depth, random_state=rs),
            {"max_depth": depth},
        ))

    # 4. Random Forests
    rf_cfg = models_cfg["random_forest"]
    for n_est in rf_cfg["n_estimators"]:
        run_name = f"random_forest_n{n_est}"
        candidates.append((
            run_name,
            "random_forest",
            RandomForestRegressor(n_estimators=n_est, random_state=rs),
            {"n_estimators": n_est},
        ))

    return candidates


def build_pipeline(model, numeric_cols: list[str], text_cols: list[str]) -> Pipeline:
    """
    Creates a scikit-learn Pipeline with ColumnTransformer:
    numeric features pass through unchanged, text/categorical features are one-hot encoded.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), text_cols),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def update_findings_tracking(findings_path: Path | str, tracking_dict: dict) -> None:
    """
    Updates only the tracking section of findings.yaml, preserving
    data_issues and monitoring sections intact.
    """
    findings_path = Path(findings_path)
    if findings_path.exists():
        with open(findings_path, "r") as f:
            doc = yaml.safe_load(f) or {}
    else:
        doc = {}

    doc["tracking"] = {
        "experiment_name": tracking_dict["experiment_name"],
        "best_run_id": tracking_dict["best_run_id"],
        "registered_model_name": tracking_dict["registered_model_name"],
        "champion_version": int(tracking_dict["champion_version"]),
    }

    with open(findings_path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def run_training() -> dict:

    params = load_params()

    # Load cleaned training data
    clean_path = ROOT / params["paths"].get("cleaned_train", "data/processed/train_clean.csv")
    df = pd.read_csv(clean_path)

    # Compute raw train.csv SHA256 (for MLflow parameter logging as required by Handbook)
    raw_train_path = ROOT / params["paths"]["train"]
    data_sha256 = compute_sha256(raw_train_path)

    # Valid feature set and target
    num_cols = params["numeric_cols"]
    txt_cols = params["text_cols"]
    target_col = params["target"]
    features = num_cols + txt_cols

    X = df[features]
    y = df[target_col]

    test_size = params["training"]["test_size"]
    random_state = params["training"]["random_state"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # MLflow Setup
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    candidates = build_candidates(params)
    results = []

    best_non_dummy = {
        "run_id": None,
        "run_name": None,
        "val_mae": float("inf"),
        "pipe": None,
    }

    print(f"Starting MLflow experiment: {params['mlflow']['experiment_name']}")
    print(f"Tracking URI: {params['mlflow']['tracking_uri']}")
    print(f"Training and logging {len(candidates)} candidate runs...\n")

    for run_name, model_type, estimator, model_params in candidates:
        with mlflow.start_run(run_name=run_name) as run:
            pipe = build_pipeline(estimator, num_cols, txt_cols)
            pipe.fit(X_train, y_train)

            preds = pipe.predict(X_val)
            val_mae = float(mean_absolute_error(y_val, preds))

            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            for k, v in model_params.items():
                mlflow.log_param(k, v if v is not None else "None")

            # Log metric
            mlflow.log_metric("val_mae", val_mae)

            # Log model artifact with cloudpickle format (safe for MLflow 3.x)
            try:
                mlflow.sklearn.log_model(
                    pipe,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                )
            except Exception:
                mlflow.sklearn.log_model(pipe, artifact_path="model")

            results.append({
                "run_name": run_name,
                "model_type": model_type,
                "val_mae": val_mae,
                "run_id": run.info.run_id,
            })

            # Track best non-dummy model
            if model_type != "dummy" and val_mae < best_non_dummy["val_mae"]:
                best_non_dummy["run_id"] = run.info.run_id
                best_non_dummy["run_name"] = run_name
                best_non_dummy["val_mae"] = val_mae
                best_non_dummy["pipe"] = pipe

    # 1. Register best model in MLflow Registry
    reg_name = params["mlflow"]["registered_model_name"]
    best_run_id = best_non_dummy["run_id"]
    model_uri = f"runs:/{best_run_id}/model"

    print(f"\nRegistering best model ({best_non_dummy['run_name']}) in Model Registry: '{reg_name}'...")
    mv = mlflow.register_model(model_uri, reg_name)
    champion_version = int(mv.version)

    # 2. Assign champion alias to Version 1
    client = MlflowClient()
    client.set_registered_model_alias(reg_name, "champion", str(champion_version))
    print(f"Set alias '@champion' on {reg_name} version {champion_version}")

    # 3. Export champion model to models/model.joblib
    model_output_rel = params["paths"].get("model_output", "models/model.joblib")
    model_output_path = ROOT / model_output_rel
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_non_dummy["pipe"], model_output_path)
    print(f"Exported model pipeline to: {model_output_path}")

    # 4. Update reports/findings.yaml tracking section
    findings_rel = params["paths"].get("findings", "reports/findings.yaml")
    findings_path = ROOT / findings_rel
    tracking_dict = {
        "experiment_name": params["mlflow"]["experiment_name"],
        "best_run_id": best_run_id,
        "registered_model_name": reg_name,
        "champion_version": champion_version,
    }
    update_findings_tracking(findings_path, tracking_dict)
    print(f"Updated tracking section in {findings_path}")

    return {
        "results": results,
        "best_run": best_non_dummy,
        "champion_version": champion_version,
    }


if __name__ == "__main__":
    train_summary = run_training()
    results = train_summary["results"]
    best = train_summary["best_run"]

    print("\n" + "=" * 65)
    print("                TRAINING SUMMARY (10 RUNS)                ")
    print("=" * 65)
    print(f"{'Run Name':<25} {'Model Type':<18} {'Val MAE (GB)':<12}")
    print("-" * 65)
    for r in results:
        marker = " <-- BEST" if r["run_id"] == best["run_id"] else ""
        print(f"{r['run_name']:<25} {r['model_type']:<18} {r['val_mae']:<12.4f}{marker}")
    print("=" * 65)
    print(f"Champion Model: {best['run_name']}")
    print(f"Best Run ID:    {best['run_id']}")
    print(f"Best Val MAE:   {best['val_mae']:.4f} GB")
    print(f"Model File:     models/model.joblib")
    print(f"Findings:       reports/findings.yaml updated")
    print("=" * 65)
