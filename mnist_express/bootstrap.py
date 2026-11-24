"""Racine de composition : le seul endroit qui choisit les adaptateurs.

Streamlit et les scripts demandent ici des services prêts à l'emploi.
Changer de stockage (base de données au lieu de JSON, registre de modèles au lieu de joblib) ne touche que ce fichier et un nouvel adaptateur.
"""
from dataclasses import dataclass
from pathlib import Path
from .adapters.clock import utc_now_iso
from .adapters.joblib_model_repository import JoblibModelRepository
from .adapters.json_usage_stats_repository import JsonUsageStatsRepository
from .config import Settings
from .core.use_cases import GetUsageStats, PredictDigit, RecordFeedback
from .inference import expected_n_features

@dataclass(frozen=True)
class AppServices:
    settings: Settings
    predict_digit: PredictDigit
    record_feedback: RecordFeedback
    get_usage_stats: GetUsageStats
    model_n_features: int | None

def build_model_repository(settings: Settings) -> JoblibModelRepository:
    return JoblibModelRepository(settings.model_dir, settings.model_filename)

def build_usage_stats_repository(settings: Settings) -> JsonUsageStatsRepository:
    return JsonUsageStatsRepository(
        Path(settings.monitoring_dir) / settings.monitoring_filename
    )

def build_services(settings: Settings | None = None) -> AppServices:
    settings = settings or Settings()
    classifier = build_model_repository(settings).load()
    usage_stats = build_usage_stats_repository(settings)

    return AppServices(
        settings=settings,
        predict_digit=PredictDigit(
            classifier=classifier,
            settings=settings,
            usage_stats=usage_stats,
            clock=utc_now_iso,
        ),
        record_feedback=RecordFeedback(usage_stats),
        get_usage_stats=GetUsageStats(usage_stats),
        model_n_features=expected_n_features(classifier),
    )