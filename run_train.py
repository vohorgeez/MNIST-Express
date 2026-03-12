from mnist_express.data import load_data
from mnist_express.train import train_knn_pipeline
from mnist_express.config import Settings
from mnist_express.persistence import save_model

def main():
    settings = Settings()

    # Dataset MNIST -> 784 features
    X_train, X_test, y_train, y_test = load_data(
        test_size=0.2,
        random_state=42,
    )

    pipe, fit_time, pred_time, acc, explained, report = train_knn_pipeline(
        X_train,
        y_train,
        X_test,
        y_test,
        settings,
    )

    save_path = save_model(pipe, settings.model_filename, settings)

    print(f"Accuracy: {acc:.4f}")
    print(f"Fit time: {fit_time:.6f} sec")
    print(f"Predict bench time: {pred_time:.6f} sec")
    print(f"Saved model to: {save_path}")
    print(f"Bench QPS: {report['timing_sec']['queries_per_sec']:.2f}")

    if explained is not None:
        print(f"PCA components kept: {len(explained)}")

if __name__ == "__main__":
    main()