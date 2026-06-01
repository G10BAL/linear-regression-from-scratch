import numpy as np

from .strategies.base import RegressionStrategy
from .strategies.normal_equation import NormalEquation
from .strategies.gradient_descent import GradientDescent
from .utils import add_bias_column

class LinearRegression:
    def __init__(self, method: str, **kwargs):
        self.strategy = self._create_strategy(method, **kwargs)
    
    @property
    def theta(self):
        return self.strategy.theta
    
    # Only for animation purposes
    @property
    def theta_history(self):
        return self.strategy.theta_history
    
    @staticmethod
    def _create_strategy(method: str, **kwargs) -> RegressionStrategy:
        strategies = {
            "normal": NormalEquation,
            "gradient_descent": GradientDescent,
        }
        if method not in strategies:
            raise ValueError(f"Unknown method: {method}")
        return strategies[method](**kwargs)
    
    def fit(self, X, y):
        self.strategy.fit(X, y)
    
    def predict(self, X, theta=None):
        if theta is None:
            theta = self.theta
        if X.ndim == 1:
            X.reshape(-1, 1)
        X = add_bias_column(X)
        return X.dot(theta)
    
    def compute_mse(self, X, y, theta=None):
        if theta is None:
            theta = self.theta
        m = len(y)
        predictions = self.predict(X, theta)
        return (1 / (2 * m)) * np.sum((predictions - y) ** 2)