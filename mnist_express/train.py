from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .config import Settings
import time
import logging
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from .metrics import accuracy

logger = logging.getLogger(__name__)

def train_knn_pipeline(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        settings: Settings
):
    # --- PCA step ---
    if settings.enable_pca:
        pca_step = PCA(
            n_components=settings.pca_n_components,
            random_state=settings.pca_random_state
        )
    else:
        pca_step = "passthrough"

    # --- KNN step ---
    knn_step = KNeighborsClassifier(
        n_neighbors=settings.knn_k,
        algorithm=settings.knn_algorithm,
        metric=settings.knn_metric
    )

    # --- Pipeline ---
    pipe = Pipeline([
        ("pca", pca_step),
        ("knn", knn_step)
    ])

    # --- Fit timing ---
    if settings.enable_timing:
        start_fit = time.perf_counter()
        pipe.fit(X_train, y_train)
        end_fit = time.perf_counter()
        fit_duration = end_fit - start_fit
    else:
        pipe.fit(X_train, y_train)
        fit_duration = 0.0

    # --- Benchmark predict timing (subset only) ---
    X_bench = X_test[:settings.bench_n_queries]

    if settings.enable_timing:
        start_predict = time.perf_counter()
        pipe.predict(X_bench)
        end_predict = time.perf_counter()
        predict_duration = end_predict - start_predict
    else:
        pipe.predict(X_bench)
        predict_duration = 0.0

    # --- Full test accuracy ---
    y_pred_full = pipe.predict(X_test)
    acc = accuracy(y_test, y_pred_full)

    # --- Explained variance ---
    if settings.enable_pca:
        explained = pipe.named_steps["pca"].explained_variance_ratio_
    else:
        explained = None

    logger.info(
        "PCA=%s | k=%d | fit=%.6f sec | predict(bench=%d)=%.6f sec | acc=%.4f",
        settings.enable_pca,
        settings.knn_k,
        fit_duration,
        settings.bench_n_queries,
        predict_duration,
        acc
    )

    return pipe, fit_duration, predict_duration, acc, explained