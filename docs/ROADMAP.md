# ROADMAP - MNIST Express v4

## Visison
Passer d'un projet exploratoire à un mini-système ML structuré, reproductible, versionné et benchmarkable.

## M1 - Architecture & Documentation

### Objectif
Stabiliser la structure du projet et formaliser les responsabilités.

### A implémenter
- Arborescence propre
- `SPEC.md` finalisé
- `ARCHITECTURE.md` (vue, modules, flux, bundle)

### Definition of Done
- Aucun code ML dans `app/`
- Chaque module a une responsabilité définie
- Les flux training/inference sont documentés
- Aucun fichier ambigu à la racine

## M2 - Training Pipeline Versionné

### Objectif
Produire un run versionné immuable à chaque entraînement

### A implémenter
- Génération `run_id`
- Création `artifacts/runs/<run_id>/`
- Sauvegarde :
    - scaler.joblib
    - pca.joblib
    - knn.joblib
    - metadata.json
    - metrics.json
- Centralisation des hyperparamètres dans `config.py`

### Definition of Done
- Deux entraînements successifs créent deux dossiers distincts
- Rechargement possible via `load_run(run_id)`
- Aucun `.fit()` en phase online
- Seed fixé et enregistré dans metadata

## M3 - Inference Contract

### Objectif
Isoler complètement la logique ML de l'UI

### A implémenter
- `ModelBundle`
- `load_run()`
- `predict_one(bundle, x)`
- Mesure latence
- Logging minimal

### Definition of Done
- Streamlit n'importe aucun joblib directement
- Inference fonctionne après redémarrage complet
- Impossible de prédire sans passer par preprocessing

## M4 - Benchmark des Variantes

### Objectif
Comparer brute, kd_tree/ball_tree et PCA optionnelle

### A implémenter
- Script de benchmark automatique
- Mesure accuracy
- Mesure latence p50/p95
- Rapport structuré dans `reports/`

### Definition of Done
- Tableau comparatif clair
- Un run marqué comme "best"
- Critères d'acceptation validés

## M5 - Tests & Robustesse

### Objectif
Garantir stabilité et non-régression

### A implémenter
- Tests preprocessing (shape, invariants)
- Tests save/load cohérence
- Test smoke inference
- Vérification absence `.fit()` en online

### Definition of Done
- Tous les tests passent
- Rechargement d'un ancien run toujours valide
- Echec explicite si artefact manquant

## M6 - Polish & Monitoring (Optionnel)

### Objectif
Rendre le projet "portfolio-ready".

### A implémenter
- Affichage latence UI
- Top-k voisins (debug)
- Compteur d'usage
- README enrichi avec schéma

### Definition of Done
- UI informative
- README compréhensible par recruteur non-tech
- Structure claire dans le repo