import autograd.numpy as np
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


def make_plot():
    plt.plot([i for i in range(0, result.num_of_iterations + 1)],
             result.min_func_value)
    plt.xlabel("Iteration")
    plt.ylabel("Value of the objective function")
    plt.title("Plot of the objective function values ​​from iteration")
    plt.title(f" alpha = 1, learning rate = 0,001, max iteration = 100")
    plt.show()


# pow(alpha, (i-1)/(n-1))


def objective_function(x, alpha):
    return np.sum(alpha * x**2.0)


new_objective_function = partial(objective_function, alpha=1)
params = params_t(0.1, 1e-6, 100, 1, 1, 1)


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


x0 = np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9])
result = gradient_descent(new_objective_function, x0, params)
print(f"Minimum found: {result.minimum}")
print(
    f"Value of the objective function - minimum: {new_objective_function(result.minimum)}"
)
print(f"Number of iterations: {result.num_of_iterations}")

plt.plot([i for i in range(0, result.num_of_iterations + 1)],
         result.min_func_value)
plt.xlabel("Iteration")
plt.ylabel("Value of the objective function")
plt.title("Plot of the objective function values ​​from iteration")
plt.title(f" alpha = 1, learning rate = 0,001, max iteration = 100")
plt.show()
