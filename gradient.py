import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from autograd import grad

obj_func_values = []


def objective_function(x, alpha):
    return np.sum(alpha * x**2)


def f_grad(x, alpha):
    return 2 * alpha * x


new_objective_function = partial(objective_function, alpha=1)
new_f_grad = partial(f_grad, alpha=1)


def gradient_descent(
    new_objective_function, x0, learning_rate=0.2, epsilon=1e-6, max_iter=200
):
    x = x0.copy()
    for i in range(max_iter):
        obj_func_values.append(new_objective_function(x=x))
        # gradient_func = grad(new_objective_function)
        gradient = new_f_grad(x)
        x = x - learning_rate * gradient
        if np.max(np.abs(learning_rate * gradient)) < epsilon:
            break
    return x, new_objective_function(x=x), i


x0 = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9])
minimum, min_func_value, num_of_iterations = gradient_descent(
    new_objective_function, x0
)
print(f"Minimum found: {minimum}")
print(f"Value of the objective function - minimum: {min_func_value}")
print(f"Number of iterations: {num_of_iterations}")

plt.plot([i for i in range(0, num_of_iterations + 1)], obj_func_values)
plt.xlabel("Iteration")
plt.ylabel("Value of the objective function")
plt.title("Plot of the objective function values ​​from iteration")
plt.show()
