MNIST Express est une mini-appli qui entraîne un classifieur k-NN sur le dataset de chiffres manuscrits MNIST (ou digits) avec scikit-learn. Le notebook mnist_knn.ipynb charge les données, normalise les features, entraîne le modèle, mesure la performance (accuracy, classification report, matrice de confusion) et met en avant quelques exemples mal classés pour analyse.

## Version déployée :
Vous pouvez accéder à l'application directement sur Streamlit à l'URL suivante :
https://mnist-express.streamlit.app/

## Prérequis : 

Python ≥3.9, pip, dépendances (scikit-learn, numpy, matplotlib, optionnel jupyter, streamlit). Installation rapide (sur terminal):
pip install jupyter scikit-learn numpy matplotlib

## Usage Notebook :

jupyter notebook ou jupyter lab dans le répertoire.
Ouvrir mnist_knn.ipynb et exécuter les cellules dans l’ordre.
Résultats attendus : précision ~97 % (digits), affichage de la matrice de confusion et de vignettes mal classées.

## UI Streamlit

Une interface légère permet de jouer avec le modèle en live.

### Installation complémentaire
pip install streamlit streamlit-drawable-canvas pillow

### Lancer l’app
streamlit run app.py

### Fonctionnalités
- Choix interactif du nombre de voisins k (slider).
- Aperçu de quelques exemples du jeu de données.
- Zone de dessin (8×8 pixels reconstruits automatiquement) pour tester ses propres chiffres manuscrits.
- Affichage instantané de la prédiction k-NN, avec visualisation optionnelle de l’image 8×8 passée au modèle.

> Astuce : pour passer sur le MNIST 28×28 depuis `fetch_openml`, adapter `app.py` (reshape 28×28 + scaler) et s’assurer que la zone de dessin redimensionne vers 28×28 avant la prédiction.