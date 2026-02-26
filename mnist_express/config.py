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

def get_settings() -> Settings:
    return Settings()