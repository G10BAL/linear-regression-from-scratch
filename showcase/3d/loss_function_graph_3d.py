from src.model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

n_samples = 100
e_std = 1000

def plot_loss_function_surface_3d():
    np.random.seed(42)
    X = np.random.randint(0, 100, size=(n_samples, 2)).astype(float)
    true_theta = np.array([2.0, 3.0])
    e = np.random.normal(20, e_std, size=n_samples)
    y = 1 + X @ true_theta + e

    lr = LinearRegression("gradient_descent", alpha=0.1, n_epochs=150)
    lr.fit(X, y)

    best_theta = lr.theta

    t1_range = np.linspace(best_theta[1] * 0.4, best_theta[1] * 2.2, 60)
    t2_range = np.linspace(best_theta[2] * 0.4, best_theta[2] * 2.2, 60)
    T1, T2 = np.meshgrid(t1_range, t2_range)
    Loss = np.zeros_like(T1)

    for i in range(T1.shape[0]):
        for j in range(T1.shape[1]):
            theta_ij = np.array([best_theta[0], T1[i, j], T2[i, j]])
            Loss[i, j] = lr.compute_mse(X, y, theta_ij)

    gd_history = [best_theta * scale for scale in [2.0, 1.6, 1.3, 1.1, 1.03, 1.0]]
    gd_t1, gd_t2, gd_losses = [], [], []
    for theta_step in gd_history:
        gd_t1.append(theta_step[1])
        gd_t2.append(theta_step[2])
        gd_losses.append(lr.compute_mse(X, y, theta_step))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(T1, T2, Loss, cmap='viridis', alpha=0.5)

    ax.scatter(gd_t1, gd_t2, gd_losses,
               color='red', s=80, zorder=10, label='GD Iterations')
    ax.plot(gd_t1, gd_t2, gd_losses,
            color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.scatter([best_theta[1]], [best_theta[2]], [lr.compute_mse(X, y, best_theta)],
               color='green', s=150, zorder=11, marker='*', label='Global Minimum')

    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$\\theta_2$')
    ax.set_zlabel('MSE')
    ax.set_title('Loss Function Surface — Gradient Descent Path')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    plot_loss_function_surface_3d()