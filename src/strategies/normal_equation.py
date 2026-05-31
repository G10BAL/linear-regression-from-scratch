import numpy as np
from .base import RegressionStrategy
from ..utils import add_bias_column

class NormalEquation(RegressionStrategy):
    def __init__(self):
        self.theta = None

    def predict(self, X):
        if X.ndim == 1:
            X.reshape(-1, 1)
        X = add_bias_column(X)
        return X.dot(self.theta)
    
    def fit(self, X, y):
        if X.ndim == 1:
            X.reshape(-1, 1)
        X = add_bias_column(X)
        self.theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        self.theta.reshape(-1, 1)
