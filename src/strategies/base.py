from abc import ABC, abstractmethod

class RegressionStrategy(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X, y): ...