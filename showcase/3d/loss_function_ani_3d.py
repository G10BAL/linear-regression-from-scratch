from src.model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n_samples = 200
np.random.seed(42)

X = np.random.randint(0, 100, size=(n_samples, 2)).astype(float)
true_theta = np.array([2.0, 3.0])
e = np.random.normal(0, 50, size=n_samples)
y = 1 + X @ true_theta + e

lr = LinearRegression("gradient_descent", alpha=0.1, n_epochs=150)
lr.fit(X, y)

best_theta = lr.theta

t1_range = np.linspace(best_theta[1] - 3, best_theta[1] + 3, 60)
t2_range = np.linspace(best_theta[2] - 3, best_theta[2] + 3, 60)
T1, T2 = np.meshgrid(t1_range, t2_range)
Loss = np.zeros_like(T1)

for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        theta_ij = np.array([best_theta[0], T1[i, j], T2[i, j]])
        Loss[i, j] = lr.compute_mse(X, y, theta_ij)

ball_t1, ball_t2, ball_loss = [], [], []
for theta_step in lr.strategy.theta_history:
    ball_t1.append(theta_step[1])
    ball_t2.append(theta_step[2])
    ball_loss.append(lr.compute_mse(X, y, theta_step))

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(projection='3d')

ax.plot_surface(T1, T2, Loss, cmap='viridis', alpha=0.4)
ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
ax.set_zlabel('MSE')
ax.set_title('Gradient Descent — Loss Surface')

ball, = ax.plot([], [], [], 'ro', markersize=10, zorder=10, label='GD Iterations')
trail, = ax.plot([], [], [], 'r-', linewidth=1.5, alpha=0.6)
epoch_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=11,
                       verticalalignment='top')
ax.legend(loc='upper right')

def update(frame):
    ball.set_data([ball_t1[frame]], [ball_t2[frame]])
    ball.set_3d_properties([ball_loss[frame]])

    trail.set_data(ball_t1[:frame+1], ball_t2[:frame+1])
    trail.set_3d_properties(ball_loss[:frame+1])

    epoch_text.set_text(
        f"Epoch: {frame}\n"
        f"θ1: {ball_t1[frame]:.3f}\n"
        f"θ2: {ball_t2[frame]:.3f}\n"
        f"MSE: {ball_loss[frame]:.2f}"
    )
    return ball, trail, epoch_text

ani = animation.FuncAnimation(
    fig, update,
    frames=len(ball_t1),
    interval=80,
    blit=False
)

plt.show()
ani.save("assets/3d/loss_surface_3d.gif", writer="pillow")