"""Ports secondaires : ce dont le coeur a besoin du monde extérieur.

Ce sont des interfaces (`typing.Protocol`) qui appartiennent au coeur. Le coeur ne sait pas si le modèle vient d'un fichier joblib ou si les statistiques sont stockées en JSON : il ne connaît que ces contrats.
"""
from typing import Protocol
import numpy as np
from .model import UsageStats

class Classifier(Protocol):
    """Un classifieur déjà entraîné (k-NN seul ou pipeline scikit-learn.)"""
    classes_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

class ModelRepository(Protocol):
    """Sauvegarde et chargement du modèle entraîné."""
    def save(self, model: Classifier) -> str:
        ...

    def load(self) -> Classifier:
        ...

class UsageStatsRepository(Protocol):
    """Lecture et écriture des statistiques d'usage."""
    def load(self) -> UsageStats:
        ...

    def save(self, stats: UsageStats) -> None:
        ...