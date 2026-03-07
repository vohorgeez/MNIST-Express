import numpy as np
import pytest
from mnist_express.preprocessing import preprocess_user_drawing

from mnist_express.preprocessing import (
    to_canonical_X,
    build_preprocessor,
    fit_transform_train,
    transform_inference,
)

# Canonical shaping (STRICT)

def test_to_canonical_X_accepts_28x28():
    img = np.zeros((28, 28), dtype=np.uint8)
    X = to_canonical_X(img)
    assert X.shape == (1, 784)
    assert X.ndim == 2

def test_to_canonical_X_accepts_784_vector():
    vec = np.zeros((784,), dtype=np.float64)
    X = to_canonical_X(vec)
    assert X.shape == (1, 784)
    assert X.ndim == 2

def test_to_canonical_X_accepts_nx784():
    batch = np.zeros((10, 784), dtype=np.float64)
    X = to_canonical_X(batch)
    assert X.shape == (10, 784)
    assert X.ndim == 2

def test_to_canonical_X_rejects_invalid_shapes():
    bad_inputs = [
        np.zeros((27, 27)),
        np.zeros((1, 28, 28)),
        np.zeros((10, 28, 28)),
        np.zeros((784, 1)),
        np.zeros((28, 28, 1)),
    ]

    for bad in bad_inputs:
        with pytest.raises(ValueError):
            to_canonical_X(bad)

# Preprocessing behavior (PCA off)

def test_preprocessing_pca_off_shape_and_dtype():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(100, 784)).astype(np.float64)

    pre = build_preprocessor(pca_enabled=False)

    X_train_t = fit_transform_train(pre, X_train)

    assert X_train_t.shape == (100, 784)
    assert X_train_t.dtype == np.float64

def test_inference_transform_pca_off_user_input_ok():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 784)).astype(np.float64)

    pre = build_preprocessor(pca_enabled=False)
    _ = fit_transform_train(pre, X_train)

    user_img = np.zeros((28, 28), dtype=np.uint8)
    X_user_t = transform_inference(pre, user_img)

    assert X_user_t.shape == (1, 784)
    assert X_user_t.dtype == np.float64

# Preprocessing behavior (PCA on)

def test_preprocessing_pca_on_shape_and_dtype():
    rng = np.random.default_rng(123)
    X_train = rng.normal(size=(120, 784)).astype(np.float64)

    n_components = 40
    pre = build_preprocessor(pca_enabled=True, n_components=n_components)

    X_train_t = fit_transform_train(pre, X_train)

    assert X_train_t.shape == (120, n_components)
    assert X_train_t.dtype == np.float64

def test_inference_transform_pca_on_user_input_ok():
    rng = np.random.default_rng(999)
    X_train = rng.normal(size=(80, 784)).astype(np.float64)

    n_components = 25
    pre = build_preprocessor(pca_enabled=True, n_components=n_components)
    _ = fit_transform_train(pre, X_train)

    user_flat = np.zeros((784,), dtype=np.float64)
    X_user_t = transform_inference(pre, user_flat)

    assert X_user_t.shape == (1, n_components)
    assert X_user_t.dtype == np.float64

# Builder guards

def test_build_preprocessor_requires_n_components_when_pca_enabled():
    with pytest.raises(ValueError):
        build_preprocessor(pca_enabled=True, n_components=None)

# Preprocess

def test_preprocess_empty_canvas():
    # arrange
    empty = np.zeros((100, 100), dtype=np.uint8)

    # act
    img = preprocess_user_drawing(empty)

    # assert
    assert isinstance(img, np.ndarray)
    assert img.shape == (28, 28)
    assert np.sum(img) == 0

def test_preprocess_returns_28x28():
    # arrange
    img_in = np.zeros((120, 80), dtype=np.uint8)
    img_in[40:80, 20:40] = 255

    # act
    img = preprocess_user_drawing(img_in)

    # assert
    assert isinstance(img, np.ndarray)
    assert img.shape == (28, 28)

def test_preprocess_offcenter_digit():
    # arrange
    img_in = np.zeros((100, 100), dtype=np.uint8)
    img_in[10:25, 10:25] = 255 # potit carré en haut à gauche

    # act
    img = preprocess_user_drawing(img_in)

    # centre de masse approximatif
    ys, xs = np.where(img > 0)
    cy = ys.mean()
    cx = xs.mean()

    # assert : proche du centre de l'image
    assert abs(cx - 14) < 4
    assert abs(cy - 14) < 4