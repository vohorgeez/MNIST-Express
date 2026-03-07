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
        predict_duration: float
    """
    if settings.enable_timing:
        start = time.perf_counter()
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        end = time.perf_counter()
        predict_duration = end - start
    else:
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        predict_duration = 0.0

    if predict_duration > 0:
        logger.info(
            "queries = %d | n_features = %d | k = %d | metric = %s | algorithm = %s | predict_duration = %.6f sec",
            X.shape[0],
            X.shape[1],
            settings.knn_k,
            settings.knn_metric,
            settings.knn_algorithm,
            predict_duration,
        )
    else:
        logger.info(
            "queries = %d | n_features = %d | k = %d | metric = %s | algorithm = %s",
            X.shape[0],
            X.shape[1],
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