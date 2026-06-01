from src.model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Variables
model = "normal"

theta_0 = -5
theta_1 = 1
theta_2 = 5
e_std = 20

def plot_3d_graph():
    np.random.seed(42)
    # Initialize the model
    lr = LinearRegression(model)
    X = np.random.randint(0, 100, size=(50, 2))
    e = np.random.normal(0, e_std, size=(50))
    y = theta_1 * X[:, 0] + theta_0 + theta_2 * X[:, 1] + e
    lr.fit(X, y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y)

    x0_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x1_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)
    x0_grid, x1_grid = np.meshgrid(x0_range, x1_range)

    y_grid = lr.theta[0] + lr.theta[1] * x0_grid + lr.theta[2] * x1_grid

    ax.plot_surface(x0_grid, x1_grid, y_grid, color='red', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    plot_3d_graph()