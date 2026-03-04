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

def accuracy_per_class(y_true, y_pred, labels=None) -> dict[int, float]:
    if labels is None:
        labels = range(10)
    results = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for c in labels:
        total = np.sum(y_true == c)
        if total == 0:
            accuracy = np.nan
        else:
            correct = np.sum((y_true == c) & (y_pred == c))
            accuracy = correct / total
        results[c] = accuracy
    return results