import autograd.numpy as np
from math import log10
from functools import partial
import matplotlib.pyplot as plt
from autograd import grad


class optim_result_t:
    def __init__(self, minimum, min_func_value, num_of_iterations):
        self.minimum = minimum
        self.min_func_value = min_func_value
        self.num_of_iterations = num_of_iterations


class params_t:
    def __init__(self, learning_rate, epsilon, max_iter, warunek1, warunek2, warunek3):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.warunek1 = warunek1
        self.warunek2 = warunek2
        self.warunek3 = warunek3


def make_plot(result):
    # styles = ['-o', '-x', '--']
    for idx, result in enumerate(results):
        plt.plot(
            [i for i in range(0, result.num_of_iterations + 1)],
            result.min_func_value,
            label=f"Result {idx+1}",
        )

    # plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value of the objective function")
    plt.title("Plot of the objective function values from iterations")
    plt.title(f"Alpha = 1, learning rate = 0.001, max iteration = 100")
    plt.legend()
    plt.show()


def objective_function(x, alpha):
    n = len(x)
    i = np.arange(n) + 1
    if (n - 1) != 0:
        x0 = (alpha ** ((i - 1) / (n - 1))) * (x**2)
    else:
        x0 = x
    return x0.sum()


# objective_functions = [partial(objective_function, alpha=1.0), partial(
#     objective_function, alpha=10.0)], partial(objective_function, alpha=100.0)
objective_function_1 = partial(objective_function, alpha=1.0)
objective_function_10 = partial(objective_function, alpha=10.0)
objective_function_100 = partial(objective_function, alpha=100.0)


def gradient_descent(new_objective_function, x0, params) -> optim_result_t:
    x = x0.copy()
    obj_func_values = []
    for i in range(params.max_iter):
        obj_func_values.append(new_objective_function(x=x))
        gradient_func = grad(new_objective_function)
        gradient = gradient_func(x)
        if np.linalg.norm(gradient) <= params.epsilon:
            break
        x = x - params.learning_rate * gradient
        if (
            params.warunek1 == 1
            and np.max(np.abs(params.learning_rate * gradient)) < params.epsilon
        ):
            break
    return optim_result_t(x, obj_func_values, i)


x0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
params = [
    params_t(0.1, 1e-6, 100, 1, 1, 1),
    params_t(0.01, 1e-6, 100, 1, 1, 1),
    params_t(0.001, 1e-6, 100, 1, 1, 1)
]
results = [
    # gradient_descent(objective_functions[0], x0, params[0]),
    # gradient_descent(objective_functions[1], x0, params[1]),
    # gradient_descent(objective_functions[2], x0, params[2]),

    gradient_descent(objective_function_1, x0, params[0]),
    gradient_descent(objective_function_10, x0, params[1]),
    gradient_descent(objective_function_100, x0, params[2]),
]

# print(f"Minimum found: {results[0].minimum}")
# print(
#     f"Value of the objective function - minimum: {objective_function_1(results[0].minimum)}"
# )
# print(f"Number of iterations: {results[0].num_of_iterations}")

make_plot(results)
