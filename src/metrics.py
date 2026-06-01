import numpy as np
from scipy import stats
from .utils import add_bias_column

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

def t_statistics(X, y, theta):
    X_b = add_bias_column(X)
    residuals = y - X_b.dot(theta)
    s2 = np.sum(residuals ** 2) / (len(y) - len(theta))
    var_theta = s2 * np.linalg.inv(X_b.T @ X_b).diagonal()
    return theta / np.sqrt(var_theta)

def p_values(X, y, theta):
    t_stats = t_statistics(X, y, theta)
    return 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y) - len(theta)))

def summary(X, y, theta):
    y_pred = add_bias_column(X).dot(theta)
    t_stats = t_statistics(X, y, theta)
    pvals = p_values(X, y, theta)

    print(f"{'':>10} {'theta':>10} {'t-stat':>10} {'p-value':>10} {'significant':>12}")
    for i, (t, ts, p) in enumerate(zip(theta, t_stats, pvals)):
        name = f'theta_{i}'
        sig = "Yes" if p < 0.05 else "No"
        print(f"{name:>10} {t:>10.4f} {ts:>10.4f} {p:>10.4f} {sig:>12}")
    print(f"\nR² = {r2(y, y_pred):.6f}")
    print(f"MSE = {mse(y, y_pred):.6f}")