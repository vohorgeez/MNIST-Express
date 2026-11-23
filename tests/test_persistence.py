"""Tests de caractérisation de la persistance du modèle"""

from pathlib import Path
import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier
from mnist_express.config import Settings
from mnist_express.persistence import load_model, save_model

def _tiny_model() -> KNeighborsClassifier:
    X = np.array([[0.1, 0.0], [0.1, 0.0], [1.0, 1.0], [0.9, 1.0]])
    y = np.array([0, 0, 1, 1])
    return KNeighborsClassifier(n_neighbors=3).fit(X, y)

def test_save_then_load_round_trip_predicts_identically(tmp_path):
    settings = Settings(model_dir=str(tmp_path / "models"))
    model = _tiny_model()
    X_check = np.array([[0.05, 0.0], [0.95, 1.0], [0.5, 0.5]])

    save_model(model, "model.joblib", settings)
    reloaded = load_model("model.joblib", settings)

    np.testing.assert_array_equal(reloaded.predict(X_check), model.predict(X_check))
    np.testing.assert_allclose(
        reloaded.predict_proba(X_check), model.predict_proba(X_check)
    )

def test_save_creates_missing_directory_and_returns_path(tmp_path):
    model_dir = tmp_path / "does" / "not" / "exist"
    settings = Settings(model_dir=str(model_dir))

    path = save_model(_tiny_model(), "model.joblib", settings)

    assert (model_dir / "model.joblib").is_file()
    assert Path(path) == model_dir / "model.joblib"

def test_load_missing_model_raises_file_not_found(tmp_path):
    settings = Settings(model_dir=str(tmp_path))

    with pytest.raises(FileNotFoundError):
        load_model("absent.joblib", settings)