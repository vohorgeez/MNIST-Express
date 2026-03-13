# MNIST Express

MNIST Express est une mini-application qui entraine un classifieur k-NN sur le dataset MNIST 28x28. Le notebook `mnist_knn.ipynb` charge les donnees, normalise les features, entraine deux variantes de k-NN (standard et PCA), mesure la performance (accuracy, classification report, matrice de confusion) et exporte les artefacts `model_knn_best.joblib` et `model_knn_pca.joblib` consommes par la mini-app Streamlit.

## Version deployee

L'application Streamlit est accessible ici : https://mnist-express.streamlit.app/

## Nouveautes v4 - benchmark et inference robuste

La version v4 introduit une instrumentation plus proche d'un contexte production :

- **Benchmark des algorithmes k-NN** (`bute`, `kd_tree`, `ball_tree`)
- **Benchmark du batching d'inférence**
- **Monitoring simple des usages**
- **Logs structurés**
- **Mini instrumentation dans l'app Streamlit**

### Résultat du benchmark des algorithmes

Un benchmark comparatif a été réalisé sur un sous-ensemble borné du dataset MNIST afin de comparer les performances des différentes stratégies de recherche de voisins.

Résultat observé :

Algorithme  Accuracy    Predict time    QPS
brute       ~0.949      ~4.6 s          ~110
kd_tree     ~0.949      ~9.1 s          ~56
ball_tree   ~0.949      ~7.2 s          ~70

Conclusion :
- les trois algorithmes produisent **la même accuracy**
- sur MNIST (784 dimensions), **`brute` est le plus performant**
- les structures d'indexation spatiale (`kd_tree`, `ball_tree`) perdent leur avantage en haute dimension

Le modèle retenu par défaut est donc :
algorithm = "brute"

### Impact du batching d'inférence

Le batching améliore fortement le débit d'inférence :

Batch size  Queries / sec
1           ~30
64          ~1100
256         ~1580
1024        ~1750

Le batching est donc activé dans le pipeline d'inférence pour améliorer les performances en usage réel.

## Nouveautes v3 - pipeline digits & double modele

- **Deux modeles embarques** : la barre laterale permet de charger instantanement `model_knn_best.joblib` (precision maximale) ou `model_knn_pca.joblib` (projection PCA 95% pour des predictions plus rapides).
- **Pipeline canvas -> digits aligne** : recadrage + centrage automatique du trace, inversion du contraste puis resize en 8x8 avant remise a l'echelle [0,16] (equivalente a `load_digits`). Le meme code est reutilise dans le notebook pour garantir la parite batch/app.
- **Outils de debug visuel** : echantillon des premiers digits affiche au dessus du canvas et checkboxes pour inspecter l'image 8x8 avant/apres passage dans le scaler.
- **UX inference affinee** : slider top-k (1 -> 10) pour filtrer les predictions interessantes, memorisation de la derniere inference (session state) pour rejouer l'analyse sans redessiner et affichage clair des probabilites ordonnees.

## Nouveautes v2 - UX interactive

La v2 introduisait :

- Une mini-app Streamlit avec canvas de dessin integre (via `streamlit-drawable-canvas`).
- Un premier pipeline de pre-traitement unifie (resize 28x28, niveaux de gris, inversion optionnelle, normalisation dans [0,1]).
- Un bouton `Predire` pour declencher l'inference en direct et afficher le label predit ainsi que les probabilites top-k.
- Des controleurs pour explorer rapidement plusieurs variantes k-NN.
- Une inference deja fluide pour des experiments rapides.

## Pipeline de pre-traitement (v3)

1. Recuperation de l'image RGBA du canvas puis passage en niveaux de gris.
2. Inversion (noir sur fond blanc -> blanc sur fond noir) pour coller a la representation `load_digits`.
3. Recadrage sur la bounding box du chiffre, padding pour le recentrer puis resize exact en 8x8 (PIL `Image.BOX`).
4. Conversion `float32`, remise a l'echelle dans [0,16] (meme dynamique que `load_digits`) puis flatten en vecteur 1x64.
5. Passage dans le scaler + k-NN deja entraine (mode precision ou PCA).

Cette chaine est partagee entre la mini-app et le notebook afin de garantir des predictions coherentes.

## Prerequis

- Python >= 3.9 et pip.
- Dependances : `scikit-learn`, `numpy`, `matplotlib`, `streamlit`, `streamlit-drawable-canvas`, `pillow`, `jupyter`, `joblib`...

Installation rapide :

```
pip install -r requirements.txt
```

## Usage Notebook

1. `jupyter notebook` ou `jupyter lab` dans ce repertoire.
2. Ouvrir `mnist_knn.ipynb`, executer toutes les cellules pour (re)generer les deux variantes k-NN et mettre a jour `model_knn_best.joblib` / `model_knn_pca.joblib`.
3. Resultats attendus : precision ~97% (digits), matrice de confusion, comparaison des temps inference best vs PCA et vignettes mal classees.

## UI Streamlit

### Lancer la mini-app

```
streamlit run app.py
```

### Fonctionnalites principales

- Apercu rapide des premiers digits du dataset pour calibrer le rendu attendu.
- Selecteur de modele (precision maximale vs rapidite PCA) dans la sidebar.
- Canvas libre 28x28 (redimensionne automatiquement) avec bouton `Predire`.
- Slider `top-k` pour afficher les classes les plus probables et probabilites detaillees.
- Checkboxes pour afficher l'image 8x8 envoyee au modele (avant et apres scaler) et comprendre l'impact du pipeline.
- Inference rapide et fluide, meme en rejouant l'analyse sur la derniere prediction.

> Astuce : le code reste prepare pour switcher vers `fetch_openml("mnist_784")` (28x28) en adaptant le reshape/le scaler dans le notebook et `preprocess_canvas_image`.
