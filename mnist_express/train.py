from copy import replace
import logging
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .config import Settings
from .metrics import (
    accuracy,
    accuracy_per_class,
    compute_confusion_matrix,
    export_metrics,
    plot_confusion_matrix,
    weakest_classes,
)

logger = logging.getLogger(__name__)


def batched_predict(
    model,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    batches = []
    for start_idx in range(0, X.shape[0], batch_size):
        end_idx = start_idx + batch_size
        batches.append(model.predict(X[start_idx:end_idx]))

    return np.concatenate(batches, axis=0)


def build_knn_pipeline(settings: Settings) -> Pipeline:
    steps = []

    if settings.enable_pca:
        steps.append((
            "pca",
            PCA(
                n_components=settings.pca_n_components,
                random_state=settings.pca_random_state,
            )
        ))

    steps.append((
        "knn",
        KNeighborsClassifier(
            n_neighbors=settings.knn_k,
            algorithm=settings.knn_algorithm,
            metric=settings.knn_metric,
        )
    ))

    return Pipeline(steps)


def benchmark_batch_sizes(
    model,
    X_bench: np.ndarray,
    batch_sizes: tuple[int, ...],
    enable_timing: bool,
) -> list[dict]:
    results = []

    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("All benchmark batch sizes must be > 0.")

        if enable_timing:
            start = time.perf_counter()
            _ = batched_predict(model, X_bench, batch_size=batch_size)
            end = time.perf_counter()
            duration = end - start
        else:
            _ = batched_predict(model, X_bench, batch_size=batch_size)
            duration = 0.0

        qps = X_bench.shape[0] / duration if duration > 0 else 0.0

        results.append({
            "batch_size": batch_size,
            "predict_duration_sec": duration,
            "queries_per_sec": qps,
        })

    return results


def train_knn_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    settings: Settings,
):
    pipe = build_knn_pipeline(settings)

    # --- Fit timing ---
    if settings.enable_timing:
        start_fit = time.perf_counter()
        pipe.fit(X_train, y_train)
        end_fit = time.perf_counter()
        fit_duration = end_fit - start_fit
    else:
        pipe.fit(X_train, y_train)
        fit_duration = 0.0

    # --- Benchmark subset ---
    n_bench = min(settings.bench_n_queries, X_test.shape[0])
    X_bench = X_test[:n_bench]

    if settings.enable_timing:
        start_predict = time.perf_counter()
        _ = pipe.predict(X_bench)
        end_predict = time.perf_counter()
        predict_duration = end_predict - start_predict
    else:
        _ = pipe.predict(X_bench)
        predict_duration = 0.0

    bench_qps = n_bench / predict_duration if predict_duration > 0 else 0.0

    # --- Batch benchmark ---
    batch_benchmark = benchmark_batch_sizes(
        model=pipe,
        X_bench=X_bench,
        batch_sizes=settings.benchmark_batch_sizes,
        enable_timing=settings.enable_timing,
    )

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
            "bench_n_queries": n_bench,
            "queries_per_sec": bench_qps,
        },
        "batch_benchmark": batch_benchmark,
        "metrics": {
            "accuracy_global": acc,
            "accuracy_per_class": acc_per_class,
            "recall_per_class": recall_per_class,
            "weakest_classes": weakest,
            "confusion_matrix": cm,
        }
    }

    tag = "pca" if settings.enable_pca else "plain"

    plot_confusion_matrix(
        cm,
        save_path=f"artifacts/metrics/confusion_{tag}.png",
    )
    export_metrics(report, f"artifacts/metrics/report_{tag}.json")
    np.savetxt(
        f"artifacts/metrics/confusion_{tag}.csv",
        cm,
        fmt="%d",
        delimiter=",",
    )

    if settings.enable_pca:
        explained = pipe.named_steps["pca"].explained_variance_ratio_
    else:
        explained = None

    logger.info(
        "train_done | pca=%s | algorithm=%s | k=%d | fit_sec=%.6f | predict_bench_sec=%.6f | bench_queries=%d | qps=%.2f | acc=%.4f",
        settings.enable_pca,
        settings.knn_algorithm,
        settings.knn_k,
        fit_duration,
        predict_duration,
        n_bench,
        bench_qps,
        acc,
    )
    logger.info("weakest_classes | values=%s", weakest)

    return pipe, fit_duration, predict_duration, acc, explained, report


def benchmark_knn_algorithms(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    settings: Settings,
) -> dict:
    results = {}

    for algorithm in settings.benchmark_algorithms:
        algo_settings = replace(settings, knn_algorithm=algorithm)

        logger.info(
            "benchmark_algorithm_start | algorithm=%s | pca=%s | k=%d",
            algo_settings.knn_algorithm,
            algo_settings.enable_pca,
            algo_settings.knn_k,
        )

        model, fit_time, predict_time, acc, explained, report = train_knn_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            settings=algo_settings,
        )

        results[algorithm] = {
            "model": model,
            "fit_time": fit_time,
            "predict_time": predict_time,
            "accuracy": acc,
            "explained_variance": explained,
            "report": report,
        }

    return results


def compare_plain_vs_pca(
    X_train,
    y_train,
    X_test,
    y_test,
    settings: Settings,
):
    results = {}

    plain_settings = replace(settings, enable_pca=False)
    model_plain, fit_plain, pred_plain, acc_plain, _, report_plain = train_knn_pipeline(
        X_train, y_train, X_test, y_test, plain_settings
    )

    results["plain"] = {
        "model": model_plain,
        "fit_time": fit_plain,
        "predict_time": pred_plain,
        "accuracy": acc_plain,
        "report": report_plain,
    }

    pca_settings = replace(settings, enable_pca=True)
    model_pca, fit_pca, pred_pca, acc_pca, explained, report_pca = train_knn_pipeline(
        X_train, y_train, X_test, y_test, pca_settings
    )

    results["pca"] = {
        "model": model_pca,
        "fit_time": fit_pca,
        "predict_time": pred_pca,
        "accuracy": acc_pca,
        "explained_variance": explained,
        "report": report_pca,
    }

    return results