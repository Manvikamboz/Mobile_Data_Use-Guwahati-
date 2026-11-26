import tempfile
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.mp238.config import load_params
from src.mp238.train import build_candidates, build_pipeline, compute_sha256


def test_load_params():
    """Verify load_params reads params.yaml and returns expected top-level keys."""
    params = load_params()
    assert isinstance(params, dict)
    assert "paths" in params
    assert "target" in params
    assert "numeric_cols" in params
    assert "text_cols" in params
    assert params["target"] == "gb_per_day"
    assert "video_minutes" in params["numeric_cols"]
    assert "phone_tier" in params["text_cols"]


def test_build_pipeline_and_predict():
    """Verify build_pipeline creates a trainable scikit-learn Pipeline on synthetic data."""
    numeric_cols = ["age", "screen_hours", "video_minutes"]
    text_cols = ["phone_tier", "circle"]

    pipeline = build_pipeline(LinearRegression(), numeric_cols, text_cols)

    # Train on small synthetic data
    X = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "screen_hours": [2.0, 3.5, 5.0, 6.0],
        "video_minutes": [60.0, 120.0, 200.0, 300.0],
        "phone_tier": ["basic", "mid", "flagship", "mid"],
        "circle": ["Urban", "Rural", "Semi-urban", "Urban"],
    })
    y = pd.Series([1.5, 2.5, 3.8, 4.2])

    pipeline.fit(X, y)
    preds = pipeline.predict(X)

    assert len(preds) == 4
    assert all(isinstance(p, (float, int)) for p in preds)


def test_build_candidates():
    """Verify build_candidates constructs candidate models including baseline and regressors."""
    params = load_params()
    candidates = build_candidates(params)

    assert len(candidates) >= 3
    model_types = {model_type for _, model_type, _, _ in candidates}
    assert "dummy" in model_types
    assert "linear_regression" in model_types
    assert "decision_tree" in model_types
    assert "random_forest" in model_types


def test_compute_sha256():
    """Verify compute_sha256 produces exact SHA-256 hexadecimal digest."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test_content_for_hash")
        temp_path = Path(f.name)

    try:
        digest = compute_sha256(temp_path)
        assert isinstance(digest, str)
        assert len(digest) == 64
        # SHA256 of "test_content_for_hash"
        import hashlib
        expected = hashlib.sha256(b"test_content_for_hash").hexdigest()
        assert digest == expected
    finally:
        temp_path.unlink()
