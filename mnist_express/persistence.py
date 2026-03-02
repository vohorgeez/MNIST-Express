import os
import joblib
from .config import Settings

def save_model(model, filename: str, settings: Settings):
    os.makedirs(settings.model_dir, exist_ok=True)
    path = os.path.join(settings.model_dir, filename)
    joblib.dump(model, path)
    return path

def load_model(filename: str, settings: Settings):
    path = os.path.join(settings.model_dir, filename)
    return joblib.load(path)