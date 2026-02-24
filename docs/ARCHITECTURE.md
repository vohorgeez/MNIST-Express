# Architecture

## Vue d'ensemble
- `app/streamlit_app.py` = UI uniquement
- `mnist_express/` = logique métier ML
- `artifacts/` = modèles + reports + monitoring

## Modules et responsabilités
- `data.py`
    - Responsabilité : chargement du dataset
    - Ne doit pas : pré-traiter le dataset
- `preprocessing.py`
    - Responsabilité : pré-traitement du dataset avec fit-scaler, fit-PCA et transform, gestion du cas PCA désactivé, garantie que transform ne fait jamais fit
    - Ne doit pas : choisir la variante de modèle utilisé
- `train.py`
    - Responsabilité : entraînement du modèle
    - Ne doit pas : contenir des valeurs hardcodées, faire de l'inférence, faire de la persistance directe
- `inference.py`
    - Responsabilité : Prédiction, mesure de latence, appel à monitoring
    - Ne doit pas : appliquer fit et toucher au dataset ou au modèle
- `persistence.py`
    - Responsabilité : save + load scaler / pca / model et gestion metadata
    - Ne doit pas : entraîner, transformer, prédire
- `metrics.py`
    - Responsabilité : Calcul des métriques offline (accuracy, confusion, precision/recall), agrégation des résultats, export éventuel
    - Ne doit pas : charger les données, entraîner le modèle, faire du logging runtime
- `monitoring.py`
    - Responsabilité : logging d'usage et d'erreurs, stockage métriques runtime et compteur d'inférence
    - Ne doit pas : calculer les métriques
- `config.py`
    - Responsabilité : hyperparamètres, choix variante, seed, chemins artefacts
    - Ne doit pas : charger les artefacts