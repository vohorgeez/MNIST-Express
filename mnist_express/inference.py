from typing import Any
import logging
import time

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


def predict_knn(
    model: Any,
    X: np.ndarray,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Predict classes for a batch of samples and return probabilities
    along with total inference duration.

    Returns:
        predictions: np.ndarray of shape (n_samples,)
        probabilities: np.ndarray of shape (n_samples, n_classes)
        predict_duration: float in seconds
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    batch_size = settings.inference_batch_size
    if batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0.")

    prediction_batches = []
    probability_batches = []

    start = time.perf_counter() if settings.enable_timing else None

    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]

        batch_predictions = model.predict(X_batch)
        batch_probabilities = model.predict_proba(X_batch)

        prediction_batches.append(batch_predictions)
        probability_batches.append(batch_probabilities)

    predict_duration = (
        time.perf_counter() - start if start is not None else 0.0
    )

    predictions = np.concatenate(prediction_batches, axis=0)
    probabilities = np.concatenate(probability_batches, axis=0)

    if predict_duration > 0:
        logger.info(
            "predict_done | queries=%d | n_features=%d | batch_size=%d | k=%d | metric=%s | algorithm=%s | predict_duration_sec=%.6f",
            X.shape[0],
            X.shape[1],
            batch_size,
            settings.knn_k,
            settings.knn_metric,
            settings.knn_algorithm,
            predict_duration,
        )
    else:
        logger.info(
            "predict_done | queries=%d | n_features=%d | batch_size=%d | k=%d | metric=%s | algorithm=%s",
            X.shape[0],
            X.shape[1],
            batch_size,
            settings.knn_k,
            settings.knn_metric,
            settings.knn_algorithm,
        )

    return predictions, probabilities, predict_duration


def build_prediction_summary(
    model: Any,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    sample_index: int = 0,
    top_k: int = 3,
) -> dict[str, object]:
    """
    Build a structured summary for one prediction.

    Returns:
        {
            "predicted_class": int,
            "confidence": float,
            "top_k": list[tuple[int, float]],
        }
    """
    if predictions.ndim != 1:
        raise ValueError("predictions must be a 1D array.")

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D array.")

    if not (0 <= sample_index < len(predictions)):
        raise IndexError("sample_index out of range.")

    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    predicted_class = int(predictions[sample_index])
    probability_row = probabilities[sample_index]

    class_labels = model.classes_
    predicted_index = int(np.where(class_labels == predicted_class)[0][0])
    confidence = float(probability_row[predicted_index])

    top_indices = np.argsort(probability_row)[::-1][:top_k]
    top_predictions = [
        (int(class_labels[i]), float(probability_row[i]))
        for i in top_indices
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k": top_predictions,
    }