import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Canonical shape enforcement (STRICT)

def to_canonical_X(X):
    """
    Convert input to canonical shape (n, 784).

    Accepted:
        (28, 28)
        (784,)
        (n, 784)

    Returns:
        np.ndarray shape (n, 784)

    Raises:
        ValueError if shape is invalid
    """

    X = np.asarray(X)

    # Case 1: single image 28x28
    if X.shape == (28, 28):
        return X.reshape(1, 784)

    # Case 2: single flattened image
    if X.shape == (784,):
        return X.reshape(1, 784)
    
    # Case 3: batch already canonical
    if X.ndim == 2 and X.shape[1] == 784:
        return X
    
    raise ValueError(f"Invalid input shape {X.shape}. Expected (28, 28), (784,), or (n,784).")

# Preprocessor builder

def build_preprocessor(pca_enabled=False, n_components=None):
    """
    Build preprocessing pipeline:
        StandardScaler (always)
        + optional PCA

    Returns:
        sklearn Pipeline
    """

    steps = []

    # Always scale
    steps.append(("scaler", StandardScaler()))

    # Optional PCA
    if pca_enabled:
        if n_components is None:
            raise ValueError("n_components must be specified if PCA is enabled.")
        steps.append(("pca", PCA(n_components=n_components)))
    
    return Pipeline(steps)

# Train-time preprocessing

def fit_transform_train(preprocessor, X_train):
    """
    Fit preprocessor on training data and transform.

    Returns:
        X_train_transformed (float64)
    """

    X_train = to_canonical_X(X_train)

    X_t = preprocessor.fit_transform(X_train)

    return X_t.astype(np.float64)

# Inference-time preprocessing
def transform_inference(preprocessor, X):
    """
    Transform input data using fitted preprocessor.

    Returns:
        X_transformed (float64)
    """

    X = to_canonical_X(X)

    X_t = preprocessor.transform(X)

    return X_t.astype(np.float64)