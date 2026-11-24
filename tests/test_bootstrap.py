"""Test d'intégration de la racine de composition, avec les vrais adaptateurs.

Modèle jobllib et statistiques JSON sont écrits dans un dossier temporaire : on vérifie que le coeur et les adaptateurs s'assemblent correctement.
"""
import pytest
from mnist_express.adapters.joblib_model_repository import JoblibModelRepository
from mnist_express.bootstrap import build_services
from mnist_express.config import Settings
from mnist_express.monitoring import load_usage_stats
from test_prediction_flow import fitted_model, vertical_bar

def _settings(tmp_path) -> Settings:
    return Settings(
        model_dir=str(tmp_path / "models"),
        monitoring_dir=str(tmp_path / "monitoring"),
    )

def test_services_predict_and_persist_usage_end_to_end(tmp_path):
    settings = _settings(tmp_path)
    JoblibModelRepository(settings.model_dir, settings.model_filename).save(fitted_model())

    services = build_services(settings)
    prediction = services.predict_digit.execute(vertical_bar(x0=135))
    services.record_feedback.execute(is_error=False)

    assert prediction.predicted_class == 1
    assert services.model_n_features == 784
    stats = load_usage_stats(settings) # relu depuis le fichier JSON
    assert stats.total_predictions == 1
    assert stats.total_feedback == 1
    assert services.get_usage_stats.execute() == stats

def test_missing_model_fails_with_an_explicit_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        build_services(_settings(tmp_path))