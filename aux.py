# auxiliary functions

import numpy as np
import torch

def y_transform(yb, n_class):
    return torch.remainder(yb, n_class)


def y_transform_int(yb, n_class):
    return yb % n_class


def spherical_kmeans(x, k, max_it):
    dx, n = x.shape
    mu = np.random.normal(0, 1, (k, dx))
    mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    for it in range(max_it):
        G = np.matmul(mu, x)
        idx = np.argmax(G, axis=0)
        for i in range(k):
            if np.any(idx == i):
                x_i = x[:, idx == i]
                x_aggr = np.mean(x_i * np.exp(5 * np.matmul(mu[i, :], x_i / np.linalg.norm(x_i, ord=2))), axis=1)
                mu[i, :] = x_aggr / np.linalg.norm(x_aggr)

    return mu


def plot_mnist_digit(ax, im, title=None, pad=None):
    ax.imshow(np.reshape(im, (28, 28)), cmap='gray')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if pad:
        ax.xaxis.set_label_coords(.5, pad)


class NoneTransform(object):
    ''' Does nothing to the image. To be used instead of None '''

    def __call__(self, image):
        return image


def x_inv_transform(xb, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return xb * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def x_transform(xb, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return (xb - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

# def plot_x_corr(ax, x, y):
#


if __name__ == '__main__':
    pass
