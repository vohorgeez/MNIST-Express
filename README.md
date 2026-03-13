# MNIST Express

MNIST Express est une mini-application pédagogique autour du classifieur **k-Nearest Neighbors** appliqué au dataset **MNIST (28×28)**.

Le projet explore plusieurs aspects d’un pipeline ML complet :

- entraînement et évaluation d’un modèle k-NN
- comparaison des algorithmes (`brute`, `kd_tree`, `ball_tree`)
- optimisation de l’inférence par **batching**
- instrumentation simple (monitoring et logs)
- interface interactive via **Streamlit**

L’application permet de **dessiner un chiffre à la main et obtenir une prédiction en temps réel**.

## Demo

Application Streamlit :

https://mnist-express.streamlit.app/

L'utilisateur peut dessiner un chiffre sur un canvas et obtenir immédiatement :
- la **classe prédite**
- la **confiance du modèle**
- les **top-k prédictions**
- le **temps d'inférence**

## Overview

Le projet couvre un pipeline ML complet :
1. Chargement du dataset MNIST
2. Pré-traitement des images dessinées par l'utilisateur
3. Entraînement d'un modèle **k-NN**
4. Benchmark des différentes stratégies d'indexation
5. Optimisation des performances d'inférence
6. Interface interactive Streamlit
7. Monitoring simple de l'utilisation de l'application

L'objectif du projet est **pédagogique** : comprendre concrètement les implications des choix d'architecture dans un pipeline ML.

## Key Features

### Entraînement k-NN
- classification d'images MNIST (784 features)
- calcul des métriques :
    - accuracy
    - recall par classe
    - confusion matrix

### Benchmark d'algorithmes
Comparaison des stratégies de recherche de voisins :
- `brute`
- `kd_tree`
- `ball_tree`

### Batching d'inférence
Optimisation du débit d'inférence en regroupant les prédictions par lots.

### Monitoring simple
Instrumentation minimale :
- compteur d'usages
- temps moyen d'inférence
- taux d'erreur utilisateur (feedback)

### Interface interactive
Application Streamlit permettant :
- de dessiner un chiffre
- d'obtenir la prédiction en temps réel
- d'observer les métriques du modèle

## Architecture

Structure principale du projet :

```
mnist_express/
 ├── config.py          # configuration globale
 ├── data.py            # chargement dataset MNIST
 ├── preprocessing.py   # pipeline de transformation des dessins
 ├── train.py           # entraînement et benchmark k-NN
 ├── inference.py       # prédiction batchée
 ├── monitoring.py      # instrumentation simple
 ├── metrics.py         # métriques et confusion matrix
 └── persistence.py     # sauvegarde des modèles
```

Application interactive :

```
app/
 └── streamlit_app.py
```

Artefacts générés :

```
artifacts/
 ├── models/
 ├── metrics/
 └── monitoring/
```

## Benchmark Results

Un benchmark comparatif a été réalisé sur un sous-ensemble borné du dataset MNIST.

Comparaison des algorithmes k-NN :

Algorithm   Accuracy    Predict Time    QPS
brute       ~0.949      ~4.6 s          ~110
kd_tree     ~0.949      ~9.1 s          ~56
ball_tree   ~0.949      ~7.2 s          ~70

### Conclusion

Les trois algorithmes produisent **la même accuracy**, mais :
- `brute` est **le plus rapide**
- `kd_tree` et `ball_tree` n'apportent pas d'avantage dans cet espace de **784 dimensions**

Le modèle retenu par défaut est donc :

```
algorithm = "brute"
```

## Impact du batching d'inférence

Le batching améliore fortement le débit :

Batch size  Queries / sec
1           ~30
64          ~1100
256         ~1580
1024        ~1750

Le pipeline d'innférence utilise donc le **batching** pour améliorer les performances.

## Installation

Pré-requis
- Python <= 3.9
- pip

Installation :

```
pip install -r requirements.txt
```

## Training

L'entraînement et le benchmark se lancent avec :

```
python run_train.py
```

Le script :
1. charge le dataset MNIST
2. exécute le benchmark `brute / kd_tree / ball_tree`
3. sélectione le meilleur modèle
4. sauvegarde le modèle entraîné dans :

```
artifacts/models/model_knn_best.joblib
```

## Streamlit App

Pour lancer l'interface interactive :

```
streamlit run app/streamlit_app.py
```

Fonctionnalités :
- canvas de dessin
- prédiction en temps réel
- affichage des probabilités
- métriques d'inférence
- instrumentation simple

## Monitoring

Le projet inclut une instrumentation légère :
- nombre total de prédictions
- temps moyen d'inférence
- nombre de feedbacks utilisateur
- taux d'erreur utilisateur

Les statistiques sont enregistrées dans :

```
artifacts/monitoring/usage_stats.json
```

## Project Evolution

### v4 - Benchmark et instrumentation
- benchmark `brute / kd_tree / ball_tree`
- batching d'inférence
- monitoring simple
- logs structurés
- amélioration de l'architecture

### v3 - Pipeline digits
- pipeline canvas -> digits
- double modèle (standard + PCA)
- debug visuel

### v2 - Interface Streamlit
- canvas de dessin
- prédiction interactive
- top-k predictions

## License

Projet pédagogique libre d'utilisation