import numpy as np
from src.model import LinearRegression

class TestNormalEquation:
    
    def test_perfect_fit_1d(self):

        X = np.arange(0, 50).reshape(-1, 1)
        y = 2 * X[:, 0] + 1
        
        model = LinearRegression(method="normal")
        model.fit(X, y)
        
        assert model.compute_mse(X, y) < 1e-10

    def test_perfect_fit_2d(self):

        X = np.random.randint(0, 100, size=(50, 2))
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1
        
        model = LinearRegression(method="normal")
        model.fit(X, y)
        
        assert model.compute_mse(X, y) < 1e-10

    def test_theta_shape(self):

        X = np.random.rand(50, 3)
        y = np.random.rand(50)
        
        model = LinearRegression(method="normal")
        model.fit(X, y)
        
        assert model.theta.shape == (4,)

    def test_coefficients_recovery(self):
        '''theta = [7, 5, -3]'''
        X = np.random.rand(100, 2)
        y = 5 * X[:, 0] + (-3) * X[:, 1] + 7
        
        model = LinearRegression(method="normal")
        model.fit(X, y)
        
        assert abs(model.theta[0] - 7) < 1e-6   # bias
        assert abs(model.theta[1] - 5) < 1e-6
        assert abs(model.theta[2] - (-3)) < 1e-6

    def test_predict_shape(self):

        X = np.random.rand(50, 2)
        y = np.random.rand(50)
        
        model = LinearRegression(method="normal")
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (50,)

    def test_noisy_data_mse_reasonable(self):

        np.random.seed(42)
        X = np.random.rand(100, 1)
        noise = np.random.normal(0, 0.1, 100)
        y = 3 * X[:, 0] + 2 + noise
        
        model = LinearRegression(method="normal")
        model.fit(X, y)
        mse = model.compute_mse(X, y)
        
        assert 0 < mse < 0.1