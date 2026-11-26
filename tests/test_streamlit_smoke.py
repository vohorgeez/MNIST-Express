"""Smoke test de l'adaptateur Streamlit, exécuté sans navigateur (AppTest).

Le canvas est un composant JavaScript : en mode test il ne renvoie aucune image, ce qui permet de vérifier le chemin 'canvas vide'. Le modèle chargé est le vrai artefact de dépôt.
"""
from pathlib import Path
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
MODEL = ROOT / "artifacts" / "models" / "model_knn_best.joblib"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model artifact missing")

@pytest.fixture
def app(monkeypatch):
    monkeypatch.chdir(ROOT) # chemins des artefacts relatifs à la racine
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at

def test_app_starts_and_shows_model_and_metrics(app):
    assert not app.exception
    assert app.title[0].value == "MNIST Express v4"
    assert app.caption[0].value == "Modèle chargé : 784 features"
    assert [m.label for m in app.metric] == [
        "Usages totaux",
        "Feedbacks reçus",
        "Taux d'erreur utilisateur",
    ]

def test_predict_on_empty_canvas_shows_a_warning(app):
    predict = next(b for b in app.button if b.label == "Prédire")

    predict.click().run()

    assert not app.exception
    assert [w.value for w in app.warning] == ["Le canvas est vide."]