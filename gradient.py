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
    names = ["max iteration = 50",
             "max iteration = 100",
             "max iteration = 250",
             "max iteration = 1000"]
    for idx, result in enumerate(results):
        plt.plot(
            [i for i in range(0, result.num_of_iterations + 1)],
            result.min_func_value,
            label=names[idx],
        )
    # plt.plot(
    #     [i for i in range(0, result[0].num_of_iterations + 1)],
    #     result[0].min_func_value)

    # plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Value of the objective function")
    plt.title("Plot of the objective function values from iterations")
    plt.title(f"α = 1, epsilon = 1e-6, β = 0.1")
    plt.legend()
    plt.savefig("a1_rozne_iter.png")
    plt.show()


def objective_function(x, alpha):
    n = len(x)
    i = np.arange(n) + 1
    if (n - 1) != 0:
        x0 = (alpha ** ((i - 1) / (n - 1))) * (x**2)
    else:
        x0 = x
    return x0.sum()


objective_function_1 = partial(objective_function, alpha=1.0)
objective_function_10 = partial(objective_function, alpha=10.0)
objective_function_100 = partial(objective_function, alpha=100.0)


def gradient_descent(new_objective_function, x0, params) -> optim_result_t:
    x = x0.copy()
    obj_func_values = []
    if params.warunek3 == 1:
        for i in range(params.max_iter):
            obj_func_values.append(new_objective_function(x=x))
            gradient_func = grad(new_objective_function)
            gradient = gradient_func(x)
            if np.linalg.norm(gradient) <= params.epsilon and params.warunek2 == 1:
                break
            x = x - params.learning_rate * gradient
            if (
                params.warunek1 == 1
                and np.max(np.abs(params.learning_rate * gradient)) < params.epsilon
            ):
                break
        return optim_result_t(x, obj_func_values, i)
    else:
        return optim_result_t(x, obj_func_values, 0)


x0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

params = [
    params_t(0.1, 1e-6, 10, 1, 1, 1),
    params_t(0.1, 1e-6, 50, 1, 1, 1),
    params_t(0.1, 1e-6, 100, 1, 1, 1),
    params_t(0.1, 1e-6, 500, 1, 1, 1)
]
results = [
    gradient_descent(objective_function_1, x0, params[0]),
    gradient_descent(objective_function_1, x0, params[1]),
    gradient_descent(objective_function_1, x0, params[2]),
    gradient_descent(objective_function_1, x0, params[3]),
]

make_plot(results)


# print(f"Minimum found: {results[0].minimum}")
# print(
#     f"Value of the objective function - minimum: {objective_function_1(results[0].minimum)}"
# )
# print(f"Number of iterations: {results[0].num_of_iterations}")
