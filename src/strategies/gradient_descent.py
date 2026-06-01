import numpy as np
from .base import RegressionStrategy
from ..utils import add_bias_column

class GradientDescent(RegressionStrategy):
    def __init__(self, alpha=0.1, n_epochs=1000):
        self.theta = None
        self.alpha = alpha
        self.n_epochs = n_epochs
        self._X_mean = None
        self._X_std = None
        self.theta_history = [] # Only for animation purposes

    def _normalize(self, X):
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0)
        return (X - self._X_mean) / self._X_std

    def fit(self, X, y):
        m = len(y)
        X_norm = add_bias_column(self._normalize(X))
        self.theta = np.zeros(X_norm.shape[1])
        self.theta_history = []
        prev_loss = 10
        for _ in range(self.n_epochs):
            y_pred = X_norm.dot(self.theta)
            loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            if abs(prev_loss - loss) < 1e-9:
                break
            gradient = (1 / m) * X_norm.T.dot(X_norm.dot(self.theta) - y)
            prev_loss = loss
            self.theta -= self.alpha * gradient
            self.theta_history.append(self._denormalized_copy())

        self._denormalize_theta()

    def _denormalize_theta(self):
        self.theta[1:] = self.theta[1:] / self._X_std
        self.theta[0] = self.theta[0] - np.dot(self.theta[1:], np.atleast_1d(self._X_mean))

    # Only for animation purposes
    def _denormalized_copy(self):
        t = self.theta.copy()
        t[1:] = t[1:] / self._X_std
        t[0] = t[0] - np.dot(t[1:], np.atleast_1d(self._X_mean))
        return t