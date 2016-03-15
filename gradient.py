import numpy as np


def f1(x):
    return x[0] * x[0] - x[1] * x[1] * x[1]


def compute_gradient(J, theta, eps=1e-5):
    """
    :param J: Function to compute gradient
    :param theta: parameters
    :param eps: precision
    :return: gradient of J
    """
    grad = np.empty(theta.shape)
    for i in range(theta.shape[0]):
        t_theta = theta.copy()
        t_theta[i] -= eps
        low = J(t_theta)
        t_theta[i] += 2 * eps
        high = J(t_theta)
        grad[i] = (high - low) / (2 * eps)
    return grad


def check_gradient():
    real_value = np.array([10, -300])
    return np.allclose(real_value, compute_gradient(f1, np.array([5., 10.]), eps=1e-6))
