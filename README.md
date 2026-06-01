# Linear Regression from Scratch

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Tests](https://github.com/username/linear-regression-from-scratch/actions/workflows/tests.yml/badge.svg)

Implementation of linear regression from scratch using two methods — 
Normal Equation and Gradient Descent — with training process visualization.

## Demo
| 2D Gradient Descent | Loss Surface 2D |
|---------------------|-----------------|
| ![](assets/2d/Gradient_Descent_2d.gif) | ![](assets/2d/loss_surface_2d.gif) |

| 3D Gradient Descent | Loss Surface 3D |
|---------------------|-----------------|
| ![](assets/3d/Gradient_Descent_3d.gif) | ![](assets/3d/loss_surface_3d.gif) |

## Math Background

### Normal Equation
An analytical solution to the MSE minimization problem, derived by setting 
the gradient to zero:

$$\theta = (X^TX)^{-1}X^Ty$$

There is a possible problem: (X^TX)^{-1} can be [singular matrix](https://en.wikipedia.org/wiki/Singular_matrix).
I deal with this problem using [pseudoinverse matrix](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse).
You can read more about the proof [here](https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression) and [here](https://eli.thegreenplace.net/2015/the-normal-equation-and-matrix-calculus/).
### Gradient Descent

An iterative optimization of the loss function:

$$\theta := \theta - \alpha \cdot \nabla J(\theta)$$

where the MSE gradient is:

$$\nabla J(\theta) = \frac{1}{m} X^T(X\theta - y)$$

### Imlementation note

In the code I use `add_bias_column` to construct the design matrix with an explicit intercept term (bias) prepended as a column of ones. Note that some implementations and literature keep the intercept separate from the feature matrix and treat it as a standalone parameter.

### Why Gradient Descent if an exact formula exists?

| | Normal Equation | Gradient Descent |
|---|---|---|
| Complexity | $O(n^3)$ — matrix inversion | $O(k \cdot n^2)$ — k iterations |
| Large datasets | Slow / impractical | Efficient |
| Learning rate $\alpha$ | Not required | Required |
| Result | Exact solution | Approximation |
| Numerical stability | Issues if $X^TX$ is singular | More stable |

Normal Equation becomes impractical when the number of predictors $n > 10{,}000$.
Gradient Descent scales well and serves as the foundation for neural networks.

### Metrics
- **MSE** — loss function
- **$R^2$** — proportion of explained variance
- **t-statistics / p-values** — coefficient significance

## Project Structure

```
linear-regression/
├── src/
│   ├── strategies/
│   │   ├── base.py           # abstract Strategy class
│   │   ├── normal_equation.py
│   │   └── gradient_descent.py
│   ├── main.py               # main file for tests
│   ├── model.py              # LinearRegression + Factory Method
│   ├── metrics.py            # MSE, R², t-test, p-values
│   └── utils.py
├── showcase/
│   ├── 2d/                   # plots for 1 predictor
│   └── 3d/                   # plots for 2 predictors
├── tests/
│   ├── test_normal_equation.py
│   └── test_gradient_descent.py
├── assets/                   # saved plots and gifs
└── main.py
```

## Tech Stack
- Python 3.10+
- NumPy — mathematical operations
- Matplotlib — visualization
- SciPy — only used for p-values
- Pytest — testing

## Installation

```bash
git clone https://github.com/username/linear-regression-from-scratch
cd linear-regression-from-scratch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

The simplest way to try it out — run `src/main.py`, 
modifying the data to your needs:

```python
model = LinearRegression(method="normal")  # or "gradient_descent"
model.fit(X, y)
print(model.theta)
```

```bash
python3 -m src.main
```

## Showcase

For a deeper understanding of how the algorithms work, it is recommended 
to run the showcase files yourself — especially the 3D plots, 
which can be rotated interactively in matplotlib:

```bash
# 2D
python3 -m showcase.2d.2d_graph
python3 -m showcase.2d.gradient_by_frames_2d
python3 -m showcase.2d.loss_function_ani_2d

# 3D — recommended for interactive exploration
python3 -m showcase.3d.3d_graph
python3 -m showcase.3d.gradient_by_frames_3d
python3 -m showcase.3d.loss_function_ani_3d
```

## Tests

```bash
python3 -m pytest tests/ -v
```

Tests verify the mathematical correctness of both methods — accuracy of 
coefficient recovery, theta shape, and gradient descent convergence.

## Sources
- [Proof of the normal equation](https://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression)
- [Calculus of matrices](https://eli.thegreenplace.net/2015/the-normal-equation-and-matrix-calculus/)
- [Gradient descent breakdown](https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/)