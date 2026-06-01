from src.model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n_samples = 100
e_std = 1000

np.random.seed(42)
X = np.random.randint(0, 100, size=(n_samples, 5))
true_theta = np.random.randint(5, 10, size=5)
e = np.random.normal(0, e_std, size=(n_samples))
y = 1 + X @ true_theta + e

lr = LinearRegression("gradient_descent", n_epochs=100)
lr.fit(X, y)

best_theta = lr.theta

scales = np.linspace(0.4, 2.2, 300)
losses = []
distances = []
for scale in scales:
    current_theta = best_theta * scale
    losses.append(lr.compute_mse(X, y, current_theta))
    distances.append(np.linalg.norm(current_theta - best_theta) * np.sign(scale - 1))

ball_distances = []
ball_losses = []
for theta_step in lr.strategy.theta_history:
    loss_step = lr.compute_mse(X, y, theta_step)
    sign = np.sign(np.sum(theta_step - best_theta))
    dist_step = np.linalg.norm(theta_step - best_theta) * (sign if sign != 0 else 1)
    ball_distances.append(dist_step)
    ball_losses.append(loss_step)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(distances, losses, color='blue', linewidth=2, alpha=0.6, label='Loss Surface')
ax.axvline(0, color='green', linestyle=':', label='Global Minimum', alpha=0.8)
ax.set_xlabel('Distance from optimal $\\theta$')
ax.set_ylabel('MSE')
ax.set_title('Gradient Descent on Loss Surface')
ax.grid(True, linestyle=':', alpha=0.6)

ball, = ax.plot([], [], 'ro', markersize=12, zorder=5)
trail, = ax.plot([], [], 'r--', alpha=0.4)
epoch_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=11,
                     verticalalignment='top')
ax.legend(loc="upper right")

def update(frame):
    ball.set_data([ball_distances[frame]], [ball_losses[frame]])
    trail.set_data(ball_distances[:frame+1], ball_losses[:frame+1])
    epoch_text.set_text(f"Epoch: {frame}\nMSE: {ball_losses[frame]:.2f}")
    return ball, trail, epoch_text

ani = animation.FuncAnimation(
    fig, update,
    frames=len(ball_distances),
    interval=80,
    blit=True
)

plt.show()
ani.save("assets/results/loss_surface_2d.gif", writer="pillow")