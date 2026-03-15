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

## Flux d'entraînement (offline)
Objectif : produire un run versionné immuable.

Etapes:
    1. Chargement du dataset via `data.py`
    2. Split train / test avec seed fixé (`config.py`)
    3. Fit du scaler sur `X_train`
    4. Si PCA activée :
        - Fit PCA sur `X_train` déjà scalé
    5. Entraînement du k-NN selon la variante (`config.py`)
    6. Evaluation via `metrics.py`
    7. Génération d'un `run_id`
    8. Création du dossier `artifacts/runs/<run_id>/`
    9. Sauvegarde :
        - scaler
        - PCA (si activée)
        - modèle
        - metadata.json
        - metrics.json

Règle critique :
Aucun objet fit sur `X_test`.

## Flux d'inférence (online)
Objectif : prédire sans modifier les artefacts

Etapes :
    1. Sélection d'un run (`latest`, `best`, ou `run_id` explicite)
    2. Chargement cohérent via `persistence.py`
    3. Image utilisateur -> vectorisation (flatten)
    4. `scaler.transform`
    5. Si PCA activée -> `pca.transform`
    6. `model.predict`
    7. Mesure latence
    8. Logging usage / erreurs via `monitoring.py`

Règle critique :
Aucun `.fit()` autorisé.

## Contrat de chargement d'un run
`persistence.py` doit exposer une seule fonction publique du type :
- `load_run(run_id=None, alias="latest")`

Elle doit :
- Lire `metadata.json`
- Charger scaler
- Charger PCA si activée
- Charger modèle
- Retourner un bundle structuré

L'app Streamlit ne doit jamais charger un joblib directement.

## Contrat interne : `ModelBundle`

### Objectif

Un seul objet "retour" standardisé pour:
- l'inférence (Streamlit)
- les tests smoke
- les scripts de benchmark

### Contenu minimal (ce que `load_run()` doit retourner)
- `scaler`: objet sklearn (toujours présent)
- `pca`: objet sklearn ou `None`
- `model`: k-NN entraîné
- `metadata`: dict chargé depuis `metadata.json`
- `run_id` : string (redondant mais pratique)

### Règles
- Aucun fit dans ce bundle (que des objets déjà fit).
- `metadata` est la source de vérité:
    - PCA activée ou non
    - chemins relatifs
    - config du modèle
    - seed / split info
- Le bundle doit exposer un accès simple au "chemin racine du run" (soit via `metadata["paths"]`, soit un champ dédié).

## Contrat fonctionnel : API d'inférence
Pour éviter que Streamlit réimplémente le pipeline à sa sauce, `inference.py` doit offrir une fonction unique:
- `predict_one(bundle, x)` retourne un objet résultat structuré, par ex:
    - `pred_label`
    - `latency_ms`
    - `neighbors` / `distances` si mode debug activé

### Règles
- `predict_one` applique exactement l'ordre:
    1. scaler.transform
    2. pca.transform (si `bundle.pca` non-None)
    3. model.predict
- Mesure latence au plus près de la prédiction.