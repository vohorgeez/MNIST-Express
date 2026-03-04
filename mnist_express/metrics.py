from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json
from pathlib import Path

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

def compute_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = range(10)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(labels)
    )

    return cm

def weakest_classes(cm: np.ndarray, labels=None, top_k=3):
    if labels is None:
        labels = range(cm.shape[0])

    cm = np.asarray(cm)

    true_counts = cm.sum(axis=1)
    correct = np.diag(cm)

    recall = np.divide(
        correct,
        true_counts,
        out=np.full_like(correct, np.nan, dtype=float),
        where=true_counts != 0
    )

    results = {label: r for label, r in zip(labels, recall)}

    weakest = sorted(results.items(), key=lambda x: x[1])[:top_k]

    return results, weakest

def export_metrics(metrics: dict, path: str):
    path = Path(path)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def _to_jsonable(obj):
    #numpy scalars -> python scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.ndarray,)):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    # plain float nan
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def export_metrics(metrics: dict, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = _to_jsonable(metrics)

    with open(path, "W", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)