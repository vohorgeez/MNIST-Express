from dataclasses import dataclass

@dataclass
class Settings:
    # --- KNN core ---
    knn_k: int = 3
    knn_metric: str = "minkowski"
    knn_algorithm: str = "brute"

    # --- Timing / benchmark ---
    enable_timing: bool = True
    bench_n_queries: int = 512
    benchmark_algorithms: tuple[str, ...] = ("brute", "kd_tree", "ball_tree")
    benchmark_batch_sizes: tuple[int, ...] = (16, 64, 256, 512)

    # --- Inference batching ---
    inference_batch_size: int = 256

    # --- Persistence ---
    model_dir: str = "artifacts/models"
    model_filename: str = "model_knn_best.joblib"

    # --- Logging ---
    log_level: str = "INFO"

    # --- PCA ---
    enable_pca: bool = False
    pca_n_components: float | int = 0.95
    pca_random_state: int = 42

    # --- Monitoring / instrumentation ---
    enable_monitoring: bool = True
    monitoring_dir: str = "artifacts/monitoring"
    monitoring_filename: str = "usage_stats.json"

    # --- User feedback / quality tracking ---
    enable_user_feedback: bool = True

def get_settings() -> Settings:
    return Settings()