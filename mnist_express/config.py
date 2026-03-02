from dataclasses import dataclass

@dataclass
class Settings:
    knn_k: int = 3
    knn_metric: str = "minkowski"
    knn_algorithm: str = "brute"
    enable_timing: bool = True
    model_dir: str = "artifacts/models"
    model_filename: str = "model_knn.joblib"
    log_level: str = "INFO"
    enable_pca: bool = False
    pca_n_components: float | int = 0.95
    pca_random_state: int = 42
    bench_n_queries: int = 2000

def get_settings() -> Settings:
    return Settings()