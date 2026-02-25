from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(test_size=0.2, random_state=42):
    """
    Load MNIST dataset (28x28 flattened to 784 features).
    Returns: X_train, X_test, y_train, y_test
    """
    dataset = fetch_openml(
        "mnist_784",
        as_frame=False,
        parser="liac-arff"
    )

    X = dataset.data
    y = dataset.target

    # --- Enforce canonical types ---
    X = X.astype(np.float64)
    y = y.astype(int)

    # --- Sanity check shapes ---
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError(f"Unexpected X shape: {X.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"Unexpected y shape: {y.shape}")
    
    # --- Train / Test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )

    return X_train, X_test, y_train, y_test