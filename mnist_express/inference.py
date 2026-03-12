from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .config import Settings
import logging
import time

logger = logging.getLogger(__name__)

def predict_knn(
        model: KNeighborsClassifier,
        X: np.ndarray,
        settings: Settings,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Predict classes and return class probabilities + inference time.

    Returns:
        predictions: np.ndarray shape (n,)
        probabilities: np.ndarray shape (n, n_classes)
        predict_duration: float (seconds)
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    
    batch_size = settings.inference_batch_size
    if batch_size <= 0:
        raise ValueError("inference_batch_size must be > 0.")
    
    predictions_batches = []
    probabilities_batches = []

    if settings.enable_timing:
        start = time.perf_counter()

    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = start_idx + batch_size
        X_batch = X[start_idx:end_idx]

        batch_predictions = model.predict(X_batch)
        batch_probabilities = model.predict_proba(X_batch)

        predictions_batches.append(batch_predictions)
        probabilities_batches.append(batch_probabilities)

    if settings.enable_timing:
        end = time.perf_counter()
        predict_duration = end - start
    else:
        predict_duration = 0.0

    predictions = np.concatenate(predictions_batches, axis=0)
    probabilities = np.concatenate(probabilities_batches, axis=0)

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

def extract_prediction_summary(
        model: KNeighborsClassifier,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        sample_index: int = 0,
        top_k: int = 3,
) -> dict:
    """
    Build a small summary for one prediction.

    Returns:
        {
            "predicted_class": int,
            "confidence": float,
            "top_k": list[tuple[int, float]]
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
    
    pred_class = int(predictions[sample_index])
    proba_row = probabilities[sample_index]

    class_labels = model.classes_
    pred_idx = int(np.where(class_labels == pred_class)[0][0])
    confidence = float(proba_row[pred_idx])

    top_indices = np.argsort(proba_row)[::-1][:top_k]
    top_predictions = [
        (int(class_labels[i]), float(proba_row[i]))
        for i in top_indices
    ]

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "top_k": top_predictions,
    }