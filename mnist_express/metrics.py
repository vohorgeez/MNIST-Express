from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred) -> float:
    return accuracy_score(y_true, y_pred)

def plot_explained_variance(explained_variance):
    cumulative = np.cumsum(explained_variance)

    plt.figure(figsize=(8,5))
    plt.plot(cumulative)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()