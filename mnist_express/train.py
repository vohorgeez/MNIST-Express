from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from .config import Settings
import time
import logging
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from .metrics import (
    accuracy,
    accuracy_per_class,
    compute_confusion_matrix,
    weakest_classes,
    export_metrics,
    plot_confusion_matrix
)

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

    acc_per_class = accuracy_per_class(y_test, y_pred_full, labels=range(10))
    cm = compute_confusion_matrix(y_test, y_pred_full, labels=range(10))
    recall_per_class, weakest = weakest_classes(cm, labels=range(10), top_k=3)

    report = {
        "pca_enabled": settings.enable_pca,
        "knn": {
            "k": settings.knn_k,
            "algorithm": settings.knn_algorithm,
            "metric": settings.knn_metric,
        },
        "timing_sec": {
            "fit": fit_duration,
            "predict_bench": predict_duration,
            "bench_n_queries": settings.bench_n_queries,
        },
        "metrics": {
            "accuracy_global": acc,
            "accuracy_per_class": acc_per_class,
            "recall_per_class": recall_per_class,
            "weakest_classes": weakest,
            "confusion_matrix": cm
        }
    }

    tag = "pca" if settings.enable_pca else "plain"
    plot_confusion_matrix(
        cm,
        save_path=f"artifacts/metrics/confusion_{tag}.png"
    )
    export_metrics(report, f"artifacts/metrics/report_{tag}.json")
    np.savetxt(f"artifacts/metrics/confusion_{tag}.csv", cm, fmt="%d", delimiter=",")

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

    logger.info("Weakest classes (by recall): %s", weakest)

    return pipe, fit_duration, predict_duration, acc, explained

def compare_plain_vs_pca(
        X_train,
        y_train,
        X_test,
        y_test,
        settings: Settings
):
    results = {}

    # --- Plain ---
    settings.enable_pca = False
    model_plain, fit_plain, pred_plain, acc_plain, _ = train_knn_pipeline(
        X_train, y_train, X_test, y_test, settings
    )

    results["plain"] = {
        "model": model_plain,
        "fit_time": fit_plain,
        "predict_time": pred_plain,
        "accuracy": acc_plain
    }

    # --- PCA ---
    settings.enable_pca = True
    model_pca, fit_pca, pred_pca, acc_pca, explained = train_knn_pipeline(
        X_train, y_train, X_test, y_test, settings
    )

    results["pca"] = {
        "model": model_pca,
        "fit_time": fit_pca,
        "predict_time": pred_pca,
        "accuracy": acc_pca,
        "explained_variance": explained
    }

    return results