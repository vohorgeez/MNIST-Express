"""Caractérisation du flux de prédiction"""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from mnist_express.config import Settings
from mnist_express.inference import extract_prediction_summary, predict_knn
from mnist_express.preprocessing import preprocess_user_drawing

def vertical_bar(x0: int = 125, width: int = 30) -> np.ndarray:
    """Un "1" : barre verticale blanche sur un canvas noir 280x280."""
    drawing = np.zeros((280, 280), dtype=np.uint8)
    drawing[50:230, x0:x0 + width] = 255
    return drawing

def ring(thickness: int = 25) -> np.ndarray:
    """Un "0" : carré évidé blanc sur un canvas noir 280x280"""
    drawing = np.zeros((280, 280), dtype=np.uint8)
    drawing[50:230, 70:210] = 255
    drawing[50 + thickness:230 - thickness, 70 + thickness:210 - thickness] = 0
    return drawing

def fitted_model() -> KNeighborsClassifier:
    """k-NN entraîné sur des dessins prétraités : trois "1", trois "0"."""
    drawings = [vertical_bar(x0) for x0 in (100, 125, 150)] + [
        ring(t) for t in (20, 25, 30)
    ]
    X = np.stack([preprocess_user_drawing(d).reshape(784) for d in drawings])
    y = np.array([1, 1, 1, 0, 0, 0])
    return KNeighborsClassifier(n_neighbors=3).fit(X.astype(np.float64), y)

def run_legacy_flow(model, settings: Settings, drawing: np.ndarray) -> dict:
    """Les étapes de `handle_prediction()`, dans le même ordre."""
    preprocessed_img = preprocess_user_drawing(drawing)
    X_input = preprocessed_img.reshape(1, 784).astype(np.float64)
    predictions, probabilities, _ = predict_knn(model=model, X=X_input, settings=settings)
    return extract_prediction_summary(
        model=model,
        predictions=predictions,
        probabilities=probabilities,
        sample_index=0,
        top_k=3,
    )

def test_vertical_bar_is_recognised_as_one():
    summary = run_legacy_flow(fitted_model(), Settings(), vertical_bar(x0=135))

    assert summary["predicted_class"] == 1
    assert summary["confidence"] == pytest.approx(1.0)
    assert summary["top_k"][0] == (1, pytest.approx(1.0))

def test_ring_is_recognised_as_zero():
    summary = run_legacy_flow(fitted_model(), Settings(), ring(thickness=22))

    assert summary["predicted_class"] == 0
    assert summary["confidence"] == pytest.approx(1.0)