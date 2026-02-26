from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .config import Settings
import time
import logging

logger = logging.getLogger(__name__)

def train_knn(
	X: np.ndarray,
	y: np.ndarray,
	settings: Settings
) -> tuple[KNeighborsClassifier, float]:
    model = KNeighborsClassifier(n_neighbors=settings.knn_k, algorithm=settings.knn_algorithm, metric=settings.knn_metric)
    if settings.enable_timing:
        start = time.perf_counter()
        model.fit(X, y)
        end = time.perf_counter()
        fit_duration = end - start
    else:
        model.fit(X, y)
        fit_duration = 0.0
    n_samples, n_features = X.shape
    logger.info("k = %d \n metric = %s \n algorithm = %s \n n_samples = %d \n n_features = %d \n fit_duration = %d", settings.knn_k, settings.knn_metric, settings.knn_algorithm, n_samples, n_features, fit_duration)
    return model, fit_duration