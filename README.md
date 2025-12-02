# MNIST Express

MNIST Express est une mini-application qui entraine un classifieur k-NN sur le dataset de chiffres manuscrits MNIST (ou digits) avec scikit-learn. Le notebook `mnist_knn.ipynb` charge les donnees, normalise les features, entraine le modele, mesure la performance (accuracy, classification report, matrice de confusion) et propose quelques exemples mal classes pour analyse.

## Version deployee

L'application Streamlit est accessible ici : https://mnist-express.streamlit.app/

## Nouveautes v2 - UX interactive

- Mini-app Streamlit avec un canvas de dessin 28x28 integre (via `streamlit-drawable-canvas`).
- Pipeline de pre-traitement unifie (resize 28x28, passage en niveaux de gris, inversion optionnelle pour corriger le fond, normalisation dans [0,1]).
- Bouton `Predire` qui declenche une inference en direct et affiche le label predit ainsi que les probabilites top-k.
- Controleurs pour k, distance et poids, afin d'experimenter rapidement plusieurs variantes k-NN.
- Inference rapide et fluide, meme en selectionnant des top-k eleves.

## Pipeline de pre-traitement

1. Redimensionnement du canvas ou de l'image utilisateur vers 28x28.
2. Conversion en niveaux de gris et normalisation dans [0,1].
3. Inversion optionnelle si le fond est clair.
4. Flatten puis passage dans le k-NN deja entraine.

Cette meme chaine est utilisee par la mini-app et par le notebook afin de garantir des predictions coherentes.

## Prerequis

- Python >= 3.9 et pip.
- Dependances : `scikit-learn`, `numpy`, `matplotlib`, `streamlit`, `streamlit-drawable-canvas`, `pillow`, `jupyter` (optionnel).

Installation rapide :

```
pip install jupyter scikit-learn numpy matplotlib streamlit streamlit-drawable-canvas pillow
```

## Usage Notebook

1. `jupyter notebook` ou `jupyter lab` dans ce repertoire.
2. Ouvrir `mnist_knn.ipynb` et executer toutes les cellules.
3. Resultats attendus : precision ~97% (digits), matrice de confusion et vignettes mal classees.

## UI Streamlit

### Lancer la mini-app

```
streamlit run app.py
```

### Fonctionnalites principales

- Canvas de dessin 28x28 avec rendu instantane.
- Bouton `Predire` pour obtenir le label et les probabilites top-k.
- Selecteurs pour k, metrique de distance et type de poids.
- Vignette de l'image normalisee envoyee au modele pour comprendre le pre-traitement.
- Vitesse d'inference adaptee a l'exploration interactive.

> Astuce : le code est prepare pour switcher facilement entre `digits` (8x8) et MNIST (28x28) depuis `fetch_openml` en adaptant le reshape et le scaler dans `app.py`.
