from src.model import LinearRegression
from src.metrics import summary
import numpy as np

def main():
    np.random.seed(42)
    # Example usage
    X = np.array([1, 2, 3, 4, 5])
    e = np.random.normal(0, 0.1, size=X.shape)
    y = np.array([2, 4, 6, 8, 10]) + e
    X_test = np.array([6, 7, 8])

    model = LinearRegression(method="gradient_descent", alpha=0.1, n_epochs=1000)
    model.fit(X, y)

    predictions = model.predict(X_test)
    print("Theta:", model.theta)
    print("Predictions:", predictions)
    print("MSE:", model.compute_mse(X, y))
    print("R²:", model.compute_r2(X, y))
    print()
    summary(X, y, model.theta)

if __name__ == "__main__":
    main()