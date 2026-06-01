from src.model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Variables
model = "gradient_descent"

e_std = 5
theta_0 = 1
theta_1 = 2
n_samples = 100

def plot_2d_graph(model):
    np.random.seed(42)
    # Initialize the model
    lr = LinearRegression(model)
    X = np.random.randint(0, 100, size=(n_samples))
    e = np.random.normal(0, e_std, size=(n_samples))
    y = theta_1 * X + theta_0 + e

    lr.fit(X, y)

    plt.scatter(X, y, label="Data")
    plt.plot(
        [X.min(), X.max()],
        [lr.theta[0] + lr.theta[1] * X.min(), lr.theta[0] + lr.theta[1] * X.max()],
        c="red",
        label=f"y = {lr.theta[0]:.2f} + {lr.theta[1]:.2f} * x"
    )
    plt.title(f"Linear Regression using {model.replace('_', ' ').title()}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    plot_2d_graph(model)