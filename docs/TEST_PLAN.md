# Tests unitaires

## U1 - Preprocessing: invariants de forme
But : garantir que transform sort le bon format
- Input : `X` shape `(n, 784)` (ou `(n, 64)` si digits)
- Output attendu:
    - PCA off -> shape identique
    - PCA on -> shape `(n, n_components)`
- DoD : asserts sur shape + dtype

## U2 - Preprocessing: pas de fit accidentel
But : `transform()` ne doit jamais fitter.
- Appeler `transform()` deux fois sur le même input -> résultats identiques
- DoD : aucun état interne ne change (à défaut, test indirect via égalité outputs)

## U3 - Persistence: layout run
But : vérifier la structure `artifacts/runs/<run_id>/`.
- Après save, vérifier présence fichiers attendus
- PCA on/off -> fichiers attendus différents
- DoD : erreurs explicites si manque

## U4 - Config: validation minimale
But : éviter une config incohérente (ex PCA on sans n_components, algo invalide).
- DoD : validation lève une exception claire

# Tests d'intégration

## I1 - Train -> Save -> Load -> Predict (round-trip)
But : un run rechargé doit prédire comme juste après training.
- Entraîner sur petit subset
- Sauvegarder run
- Recharger run
- Comparer prédictions sur un batch fixe
- DoD : prédictions identiques (ou tolérance définie si probabilités)

## I2 - No fit online (contrat)
But : empêcher `.fit()` en inference.
- Charger bundle
- Faire `predict_one()`
- DoD : aucun appel à `.fit()` (stratégie: mock/spy ou wrappers, selon framework de test)

## I3 - Variantes supportées
But : brute/kd_tree/ball_tree + PCA on/off fonctionnent.
- Paramétrer tests sur les combinaisons autorisées
- DoD : chaque combinaison produit au moins un run valide + inference ok

# Smoke tests (UI-indépendants)

## S1 - Chargement "latest"
But : `load_run(alias="latest")` marche.
- Créer 2 runs
- Charger latest
- DoD : latest correspond au dernier créé

## S2 - Predict minimal
But : une prédiction sur un seul sample marche.
- Input : un sample du dataset
- DoD : retourne label int + latency_ms

# Tests de performance (locaux, non bloquants CI)

## P1 - Latence inference p95
But : vérifier que p95 respecte le seuil.
- Faire N prédictions (ex 200)
- Mesurer latence
- DoD : p95 < seuil défini dans SPEC (100ms)

## P2 - Comparatif variantes
But : brute vs kd/ball vs PCA
- DoD : produire un tableau comparatif (artifact report)