from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .config import Settings
import logging
import time

logger = logging.getLogger(__name__)

def predict_knn(model: KNeighborsClassifier, X: np.ndarray, settings: Settings) -> tuple[np.ndarray, float]:
    if settings.enable_timing:
        start = time.perf_counter()
        prediction = model.predict(X)
        end = time.perf_counter()
        predict_duration = end - start
    else:
        prediction = model.predict(X)
        predict_duration = 0.0
    if predict_duration > 0:
        logger.info("queries = %d | n_features = %d | k = %d | metric = %s | algorithm = %s | predict_duration = %.6f sec", X.shape[0], X.shape[1], settings.knn_k, settings.knn_metric, settings.knn_algorithm, predict_duration)
    else:
        logger.info("queries = %d | n_features = %d | k = %d | metric = %s | algorithm = %s", X.shape[0], X.shape[1], settings.knn_k, settings.knn_metric, settings.knn_algorithm)
    return prediction, predict_duration