MNIST Express est une mini-appli qui entraîne un classifieur k-NN sur le dataset de chiffres manuscrits MNIST (ou digits) avec scikit-learn. Le notebook mnist_knn.ipynb charge les données, normalise les features, entraîne le modèle, mesure la performance (accuracy, classification report, matrice de confusion) et met en avant quelques exemples mal classés pour analyse.

Prérequis : Python ≥3.9, pip, dépendances (scikit-learn, numpy, matplotlib, optionnel jupyter, streamlit). Installation rapide (sur terminal):
pip install jupyter scikit-learn numpy matplotlib

Usage Notebook :
jupyter notebook ou jupyter lab dans le répertoire.
Ouvrir mnist_knn.ipynb et exécuter les cellules dans l’ordre.
Résultats attendus : précision ~97 % (digits), affichage de la matrice de confusion et de vignettes mal classées.