from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from mnist_express.train import train_knn_pipeline
from mnist_express.config import Settings

# Dataset
digits = load_digits()

X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

settings = Settings()

pipe, fit_time, pred_time, acc, explained = train_knn_pipeline(
    X_train,
    y_train,
    X_test,
    y_test,
    settings
)

print("Accuracy:", acc)