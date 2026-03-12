from mnist_express.config import Settings
from mnist_express.data import load_data
from mnist_express.persistence import save_model
from mnist_express.train import benchmark_knn_algorithms


def main():
    settings = Settings()

    X_train, X_test, y_train, y_test = load_data(
        test_size=0.2,
        random_state=42,
    )

    results = benchmark_knn_algorithms(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        settings=settings,
    )

    best_algorithm = max(
        results.items(),
        key=lambda item: item[1]["accuracy"]
    )[0]

    best_model = results[best_algorithm]["model"]
    save_path = save_model(best_model, settings.model_filename, settings)

    print("=== KNN algorithm benchmark ===")
    for algorithm, result in results.items():
        report = result["report"]
        print(
            f"{algorithm}: "
            f"acc={result['accuracy']:.4f} | "
            f"fit={result['fit_time']:.6f}s | "
            f"predict={result['predict_time']:.6f}s | "
            f"qps={report['timing_sec']['queries_per_sec']:.2f}"
        )

    print(f"Best algorithm: {best_algorithm}")
    print(f"Saved model to: {save_path}")


if __name__ == "__main__":
    main()