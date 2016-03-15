import matplotlib.pyplot as plt
import numpy as np


def display_layer(X, filename='w_layer.png'):
    """
    :param X: numpy array (N x D), contain N images to display
    :param filename: name of file to save image
    """
    d = int(np.sqrt(X.shape[1]/3.0))
    N = X.shape[0]
    im_in_line = 10
    width = im_in_line * d
    high = N/im_in_line * d
    image = np.empty((high, width, 3), dtype='uint8')
    for i in range(N/im_in_line):
        for j in range(im_in_line):
            image[i * d:(i + 1) * d, j * d:(j + 1) * d] = X[i * im_in_line + j].reshape(d, d, 3)

    plt.imshow(image, interpolation=None)
    plt.axis('off')
    plt.savefig(filename)

    plt.show()
