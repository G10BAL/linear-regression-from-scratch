from src.model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

n_samples = 100
e_std = 10
theta_0 = 1
theta_1 = 2

X = np.random.randint(0, 100, size=(n_samples))
e = np.random.normal(0, e_std, size=(n_samples))
y = theta_1 * X + theta_0 + e

fig, ax = plt.subplots()

X_plot = np.linspace(X[:].min(), X[:].max(), 100)

scatter = ax.scatter(X[:], y, alpha=0.5)
line, = ax.plot([], [], 'r-', linewidth=2)
title = ax.set_title("")

lin_r = LinearRegression(method="gradient_descent", alpha=0.1, n_epochs=100)
lin_r.fit(X, y)

def update(frame):
    theta = lin_r.theta_history[frame]
    y_line = theta[0] + theta[1] * X_plot
    line.set_data(X_plot, y_line)
    line.set_label(f"y = {theta[1]:.2f}x + {theta[0]:.2f}")
    ax.legend(loc="upper left")
    title.set_text(f"Epoch {frame}")
    return line, title

ani = animation.FuncAnimation(
    fig, update,
    frames=len(lin_r.theta_history),
    interval=50,
    blit=True
)

plt.show()
ani.save("assets/2d/Gradient_Descent_2d.gif", writer="pillow")