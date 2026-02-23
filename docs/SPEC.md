# SPEC

## Contexte
Le digit-dataset MNIST est un peu le "Hello World" du Machine Learning. C'est un passage obligé pour se former à l'intelligence artificielle et à la compréhension globale des algorithmes de classification en ML.

## Objectif
Construire une application Streamlit capable de reconnaître le chiffre tracé à la main par l'utilisateur grâce à un modèle pré-entraîné sur le dataset MNIST.

## Variantes

### Brute-force
- Chaque image 28x28 du dataset est convertie en vecteur de 784 dimensions.
- Un modèle est ainsi formé en classant chaque vecteur par son label correspondant (apprentissage supervisé)
- L'input utilisateur, après normalisation, est convertie en vecteur 784D et comparé à toutes les données du modèles
- k-NN dégage les k plus proches voisins du vecteur input et en déduit le label correspondant.

### KD/Ball Tree
Même procédé que brute force précédent mais à la différence :
- On ne compare pas l'input à l'ensemble du dataset
- On lit les données en les ordonnant en un arbre de recherche (KD Tree)
- On élague les branches inutiles
=> Opérations de recherche économisées => Gain de performance

### PCA optionnelle
Appliquer un PCA au dataset permet, à partir de la variance des features :
- de différencier les features discriminantes des features "bruit"
- et donc de réduire le nombre de dimensions.
Cette réduction peut améliorer la vitesse et parfois la performance, mais sans garantie.

## Critères d'acceptation
- Accuracy >= baseline brute-force
- p95 < 100ms
- Entraînement offline : aucun fit en production
- Seed fixé + version des dépendances figée + artefacts versionnés

## Pipeline

### Phase d'entraînement (offline)
- Chargement dataset
- Split train/test
- Fit preprocessing (scaler + PCA optionnelle)
- Entraînement k-NN
- Evaluation
- Sauvegarde des artefacts

### Phase d'inférence (online)
- Chargement des artefacts sauvegardés
- Transformation de l'image utilisateur (sans fit)
- Prédiction
- Mesure du temps d'inférence
- Logging d'usage

## Hors périmètre
- Deep Learning (CNN)
- Optimisation GPU
- Entraînement continu en production