import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from mnist_express.config import Settings
from mnist_express.inference import predict_knn, extract_prediction_summary

def test_predict_knn_returns_prediction_proba_and_duration():
    X_train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [1.0, 1.0],
        [0.9, 1.0],
    ])
    y_train = np.array([0, 0, 1, 1])

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    settings = Settings()
    X_test = np.array([[0.05, 0.0]])

    predictions, probabilities, duration = predict_knn(model, X_test, settings)

    assert predictions.shape == (1,)
    assert probabilities.shape == (1, 2)
    assert isinstance(duration, float)

def test_extract_prediction_summary_returns_expected_keys():
    X_train = np.array([
        [0.0, 0.0],
        [0.1, 0.0],
        [1.0, 1.0],
        [0.9, 1.0],
    ])
    y_train = np.array([0, 0, 1, 1])

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    settings = Settings()
    X_test = np.array([[0.05, 0.0]])

    predictions, probabilities, _ = predict_knn(model, X_test, settings)
    summary = extract_prediction_summary(model, predictions, probabilities)

    assert "predicted_class" in summary
    assert "confidence" in summary
    assert "top_k" in summary
    assert isinstance(summary["predicted_class"], int)
    assert isinstance(summary["confidence"], float)
    assert isinstance(summary["top_k"], list)