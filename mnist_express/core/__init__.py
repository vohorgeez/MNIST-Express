"""Coeur de l'application (l'intérieur de l'hexagone)

Règle de dépendance : rien ici n'importe Streamlit, joblib, ni aucun module qui lit ou écrit des fichiers. Le coeur décrit ce dont il a besoin dans `ports.py` ; les implémentations concrètes vivent dans `mnist_express.adapters`.
"""