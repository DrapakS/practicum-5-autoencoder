import numpy as np
from scipy.optimize import minimize


def sigmoid(x):
    x = x.astype(float)
    return 1.0/(1 + np.exp(-x))


def KL_div(p1, p2):
    return p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1)/(1 - p2))


def deriv_sigm(x):
    return sigmoid(x) * (1 - sigmoid(x))


def initialize(hidden_size, visible_size):
    """
        Initialize autoencoder's weights
        with small values from uniform distribution
        Parametrs:
        :param hidden_size: iterable
                            size of hidden layers
        :param visible: int
                        size of visible layers
        :return: numpy array
                 flatten array of weights (with bias unit weights in the end)
    """

    k = np.sqrt(6./(visible_size + hidden_size[0] + 1))
    W_count = visible_size * hidden_size[0]
    b_count = hidden_size[0]

    for i in range(hidden_size.shape[0] - 1):
        W_count += hidden_size[i] * hidden_size[i + 1]
        b_count += hidden_size[i + 1]

    W_count += hidden_size[-1] * visible_size
    b_count += visible_size

    params_W = (np.random.random(size=W_count) - 0.5) * 2.0 * k
    params_b = np.zeros((b_count))
    return np.concatenate((params_W, params_b))


def autoencoder_loss(theta, visible_size, hidden_size, lambda_, sparsity_param, beta, data):
    """
    :param theta: numpy array. Autoencoder weights
    :param visible_size: int, size of visible layers
    :param hidden_size: iterable, size of hidden layers
    :param lambda_: float, l2 regularization param
    :param sparsity_param: float, sparsity regularization param
    :param beta: float, coefficient of the KL divergence in sparsity regularization
    :param data: data for learn
    :return: float, cost function value
             numpy array, gradient of cost function
    """
    data = (data.transpose()).astype(float)
    m = data.shape[1]
    W_count = visible_size * hidden_size[0]
    for i in range(hidden_size.shape[0] - 1):
        W_count += hidden_size[i] * hidden_size[i + 1]
    W_count += hidden_size[-1] * visible_size

    W = theta[:W_count]
    b = theta[W_count:]

    layer_data = data

    last_param_W = 0
    last_param_b = 0
    layers = np.hstack((np.array([visible_size]), hidden_size, np.array([visible_size])))

    interm_values = []
    W_list = []

    for i in range(layers.shape[0] - 1):
        layer_params = layers[i] * layers[i + 1]
        layer_W = W[last_param_W:last_param_W + layer_params].reshape((layers[i + 1], layers[i]))
        W_list.append(layer_W)
        last_param_W += layer_params
        layer_b = b[last_param_b:last_param_b + layers[i + 1]]
        last_param_b += layers[i + 1]

        if i != 0:
            layer_data = layer_W.dot(sigmoid(layer_data)) + np.tile(layer_b, (m, 1)).transpose()
        else:
            layer_data = layer_W.dot(layer_data) + np.tile(layer_b, (m, 1)).transpose()
        interm_values.append(layer_data)

    sparsity_distr = np.tile(sparsity_param, np.min(hidden_size))
    hidden_layer_distr = np.sum(sigmoid(interm_values[np.argmin(hidden_size)]), axis=1)/float(m)

    cost = (1/(2 * float(m))) * np.sum((sigmoid(layer_data) - data) ** 2) + (float(lambda_)/2) * np.sum(W ** 2)
    cost += float(beta) * np.sum(KL_div(sparsity_distr, hidden_layer_distr))

    delta = -(data - sigmoid(layer_data)) * deriv_sigm(layer_data)
    grad_W = np.empty(0)
    grad_b = np.empty(0)

    sparsity_delta = np.tile(-sparsity_distr/ hidden_layer_distr + \
                             (1 - sparsity_distr) / (1 - hidden_layer_distr), (m, 1)).transpose()

    for i in reversed(range(layers.shape[0] - 1)):
        if i != 0:
            layer_W_grad = delta.dot(sigmoid(interm_values[i - 1]).transpose()) / m + lambda_ * W_list[i]
        else:
            layer_W_grad = delta.dot(data.transpose()) / m + lambda_ * W_list[i]
        layer_b_grad = np.sum(delta, axis=1) / m

        grad_W = np.concatenate((layer_W_grad.flatten(), grad_W))
        grad_b = np.concatenate((layer_b_grad.flatten(), grad_b))

        if i == np.argmin(layers):
            delta = (W_list[i].transpose().dot(delta) + beta * sparsity_delta) * deriv_sigm(interm_values[i - 1])
        else:
            delta = (W_list[i].transpose().dot(delta)) * deriv_sigm(interm_values[i - 1])

    grad = np.concatenate((grad_W, grad_b))
    return cost, grad


def autoencoder_fit(visible_size, hidden_size, lambda_, sparsity_param, beta, data, n_iter=1000):
    """
    :return: autoencoder's weights
    """
    theta = initialize(hidden_size, visible_size)
    opt_func = lambda x: autoencoder_loss(x, visible_size, hidden_size, lambda_, sparsity_param, beta, data)
    res = minimize(opt_func, theta, method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': n_iter})
    return res.x


def autoencoder_transform(theta, visible_size, hidden_size, layer_number, data):
    """
    :param: layer_number: number of layer with new features
    :return: transformed data with shape (data.shape[0], hidden_size[layer_number])
    """
    data = (data.transpose()).astype(float)
    m = data.shape[1]
    W_count = visible_size * hidden_size[0]
    for i in range(hidden_size.shape[0] - 1):
        W_count += hidden_size[i] * hidden_size[i + 1]
    W_count += hidden_size[-1] * visible_size

    W = theta[:W_count]
    b = theta[W_count:]

    layer_data = data

    last_param_W = 0
    last_param_b = 0
    layers = np.hstack((np.array([visible_size]), hidden_size, np.array([visible_size])))

    W_list = []

    for i in range(layers.shape[0] - 1):
        layer_params = layers[i] * layers[i + 1]
        layer_W = W[last_param_W:last_param_W + layer_params].reshape((layers[i + 1], layers[i]))
        W_list.append(layer_W)
        last_param_W += layer_params
        layer_b = b[last_param_b:last_param_b + layers[i + 1]]
        last_param_b += layers[i + 1]

        if i != 0:
            layer_data = layer_W.dot(sigmoid(layer_data)) + np.tile(layer_b, (m, 1)).transpose()
        else:
            layer_data = layer_W.dot(layer_data) + np.tile(layer_b, (m, 1)).transpose()

        if i == layer_number:
            return sigmoid(layer_data)
