from src.model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

n_samples = 100
e_std = 1000

def plot_loss_function_surface():
    np.random.seed(42)

    X = np.random.randint(0, 100, size=(n_samples, 5))
    true_theta = np.random.randint(5, 10, size=5)
    e = np.random.normal(20, e_std, size=(n_samples))
    y = 1 + X @ true_theta + e

    lr = LinearRegression("gradient_descent")
    lr.fit(X, y)
    best_theta = lr.theta  

    gd_history = [best_theta * scale for scale in [2.0, 1.6, 1.3, 1.1, 1.03, 1.0]]
    gd_distances = []
    gd_losses = []

    for theta_step in gd_history:
        loss_step = lr.compute_mse(X, y, theta_step)
        gd_losses.append(loss_step)
        
        sign = np.sign(np.sum(theta_step - best_theta))
        dist_step = np.linalg.norm(theta_step - best_theta) * (sign if sign != 0 else 1)
        gd_distances.append(dist_step)


    scales = np.linspace(0.4, 2.2, 100)
    losses = []
    distances = []

    for scale in scales:
        current_theta = best_theta * scale
        losses.append(lr.compute_mse(X, y, current_theta))
        distances.append(np.linalg.norm(current_theta - best_theta) * np.sign(scale - 1))


    plt.figure(figsize=(10, 6))

    plt.plot(distances, losses, color='blue', linewidth=2, alpha=0.6, label='Loss Function Surface')

    plt.scatter(gd_distances, gd_losses, color='red', s=50, zorder=5, label='GD Iterations (Real Data)')
    plt.plot(gd_distances, gd_losses, color='red', linestyle='--', alpha=0.5)
    plt.axvline(0, color='green', linestyle=':', label='Global Minimum', alpha=0.8)

    plt.title('Gradient Descent divergence')
    plt.xlabel('Distance from optimal $\\theta*$')
    plt.ylabel('MSE Value')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_loss_function_surface()