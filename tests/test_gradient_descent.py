import numpy as np
from src.model import LinearRegression

class TestGradientDescent:

    def test_perfect_fit_1d(self):

        X = np.arange(0, 50).reshape(-1, 1).astype(float)
        y = 2 * X[:, 0] + 1

        model = LinearRegression(method="gradient_descent", alpha=0.1, n_epochs=1000)
        model.fit(X, y)

        assert model.compute_mse(X, y) < 1e-6

    def test_perfect_fit_2d(self):

        X = np.random.randint(0, 100, size=(50, 2)).astype(float)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1

        model = LinearRegression(method="gradient_descent", alpha=0.1, n_epochs=1000)
        model.fit(X, y)

        assert model.compute_mse(X, y) < 1e-6

    def test_theta_shape(self):

        X = np.random.rand(50, 3)
        y = np.random.rand(50)

        model = LinearRegression(method="gradient_descent")
        model.fit(X, y)

        assert model.theta.shape == (4,)

    def test_coefficients_recovery(self):

        np.random.seed(42)
        X = np.random.rand(200, 2)
        y = 5 * X[:, 0] + (-3) * X[:, 1] + 7

        model = LinearRegression(method="gradient_descent", alpha=0.1, n_epochs=2000)
        model.fit(X, y)

        assert abs(model.theta[0] - 7) < 1e-2
        assert abs(model.theta[1] - 5) < 1e-2
        assert abs(model.theta[2] - (-3)) < 1e-2

    def test_predict_shape(self):

        X = np.random.rand(50, 2)
        y = np.random.rand(50)

        model = LinearRegression(method="gradient_descent")
        model.fit(X, y)

        assert model.predict(X).shape == (50,)

    def test_noisy_data_mse_reasonable(self):

        np.random.seed(42)
        X = np.random.rand(100, 1)
        noise = np.random.normal(0, 0.1, 100)
        y = 3 * X[:, 0] + 2 + noise

        model = LinearRegression(method="gradient_descent", alpha=0.1, n_epochs=1000)
        model.fit(X, y)
        mse = model.compute_mse(X, y)

        assert 0 < mse < 0.1

    def test_early_stopping(self):

        X = np.random.rand(50, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1

        model = LinearRegression(method="gradient_descent", alpha=0.1, n_epochs=10000)
        model.fit(X, y)

        assert len(model.strategy.theta_history) < 10000
