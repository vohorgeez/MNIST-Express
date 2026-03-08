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

def binarize(draw: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Binarise l'image pour détecter la masse utile.
    """
    draw = np.asarray(draw)

    if draw.ndim != 2:
        raise ValueError("Input drawing must be 2D.")
    
    return (draw > threshold).astype(np.uint8)

def useful_bounding_box(binarized: np.ndarray):
    """
    Compute bounding box of useful pixels.
    Returns (y_min, y_max, x_min, x_max) or None if empty.
    """
    ys, xs = np.where(binarized > 0)

    if len(xs) == 0:
        return None
    
    return ys.min(), ys.max(), xs.min(), xs.max()

def resize(draw: np.ndarray, bounding_box):
    """
    Crop + resize to MNIST-like square while preserving ratio.
    """
    if bounding_box is None:
        return np.zeros((28, 28), dtype=np.float32)
    
    y_min, y_max, x_min, x_max = bounding_box

    cropped = draw[y_min:y_max + 1, x_min:x_max + 1]

    h, w = cropped.shape

    scale = 20 / max(h, w)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = np.zeros((new_h, new_w), dtype=np.float32)

    # simple nearest neighbor resize
    for i in range(new_h):
        for j in range(new_w):
            src_y = int(i / scale)
            src_x = int(j / scale)
            resized[i, j] = cropped[min(src_y, h - 1), min(src_x, w - 1)]

    canvas = np.zeros((28, 28), dtype=np.float32)

    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas

def calculate_mass_center(img: np.ndarray):
    """
    Compute center of mass.
    """
    ys, xs = np.indices(img.shape)

    total = img.sum()

    if total == 0:
        return None
    
    cy = (ys * img).sum() / total
    cx = (xs * img).sum() / total

    return cy, cx

def center(img: np.ndarray, mass_center):
    """
    Recenter image using center of mass.
    """
    if mass_center is None:
        return img
    
    cy, cx = mass_center

    shift_y = int(round(14 - cy))
    shift_x = int(round(14 - cx))

    recentered = np.roll(img, shift_y, axis=0)
    recentered = np.roll(recentered, shift_x, axis=1)

    return recentered

def preprocess_user_drawing(draw: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for user drawing.
    """
    draw = np.asarray(draw)

    if draw.ndim != 2:
        raise ValueError("Drawing must be a 2D array.")
    
    binarized = binarize(draw)

    bounding_box = useful_bounding_box(binarized)

    resized = resize(draw, bounding_box)

    # normalize
    resized = resized.astype(np.float32)

    mass_center = calculate_mass_center(resized)

    recentered = center(resized, mass_center)

    return recentered