from src.model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

n_samples = 100
e_std = 10
theta_0 = 1
theta_1 = 2
theta_2 = 3

X = np.random.randint(0, 100, size=(n_samples, 2))
e = np.random.normal(0, e_std, size=(n_samples))
y = theta_1 * X[:, 0] + theta_2 * X[:, 1] + theta_0 + e

lin_r = LinearRegression(method="gradient_descent", alpha=0.01, n_epochs=1000)
lin_r.fit(X, y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
x0_grid, x1_grid = np.meshgrid(x0_range, x1_range)

ax.scatter(X[:, 0], X[:, 1], y, alpha=0.5)
surface = [ax.plot_surface(x0_grid, x1_grid,
    np.zeros_like(x0_grid), alpha=0.3, color='red')]
title = ax.set_title("")

info_text = [ax.text2D(0.05, 0.95, "", transform=ax.transAxes)]

def update(frame):
    surface[0].remove()
    theta = lin_r.theta_history[frame]
    z = theta[0] + theta[1] * x0_grid + theta[2] * x1_grid
    surface[0] = ax.plot_surface(x0_grid, x1_grid, z, alpha=0.3, color='red')
    
    info_text[0].set_text(
        f"Epoch {frame}\n"
        f"y = {theta[1]:.2f}*x1 + {theta[2]:.2f}*x2 + {theta[0]:.2f}"
    )
    return surface[0], info_text[0]

ani = animation.FuncAnimation(
    fig, update,
    frames=len(lin_r.theta_history),
    interval=25,
    blit=False
)

plt.show()
ani.save("assets/3d/Gradient_Descent_3d.gif", writer="pillow")