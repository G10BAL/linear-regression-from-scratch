import numpy as np

def add_bias_column(X : np.array) -> np.array:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    betta0 = np.ones(X.shape[0])
    betta0 = betta0.reshape(-1, 1)
    return np.concatenate((betta0, X), axis=1)