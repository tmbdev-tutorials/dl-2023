import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib.pyplot import *
from scipy.stats import kde
from sklearn import decomposition
from torchvision import datasets


device = "cuda:0"


def plot_density(data, nbins=100):
    x, y = data.T
    k = kde.gaussian_kde(data.T)
    xi, yi = np.mgrid[x.min() : x.max() : nbins * 1j, y.min() : y.max() : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap=plt.cm.BuGn_r)
    contour(xi, yi, zi.reshape(xi.shape))


def tshow(v, ax=None, **keys):
    if isinstance(v, torch.Tensor):
        v = v.cpu().detach().numpy()
    if v.ndim == 1:
        v = v.reshape(28, 28)
    if ax is not None:
        ax.imshow(v, **keys)
    else:
        imshow(v, **keys)


def showgrid(images, rows=4, cols=4, cmap=cm.gray, **kw):
    for i in range(rows * cols):
        subplot(rows, cols, i + 1)
        xticks([])
        yticks([])
        tshow(images[i], cmap=cmap, **kw)


def showrow(*args):
    for i, im in enumerate(args):
        subplot(1, len(args), i + 1)
        tshow(im)


def T(a):
    return torch.FloatTensor(a).to(device)


def N(a):
    return a.cpu().detach().numpy()


def display_components(a, l, fs=None, rows=1, row=0, yscale="log", ylim=None, figsize=(6, 6)):
    fig = plt.figure(figsize=fs)
    outer = gridspec.GridSpec(rows, 2, wspace=0.2, hspace=0.2)
    inner = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer[2 * row], wspace=0.1, hspace=0.1)
    for j in range(16):
        ax = plt.Subplot(fig, inner[j])
        image = a[j]
        if isinstance(a, (torch.Tensor, torch.cuda.FloatTensor)):
            a = N(a)
        if image.ndim == 1:
            image = image.reshape(len(image) // 28, 28)
        ax.imshow(image, cmap=cm.RdBu)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    if l is not None:
        ax = plt.Subplot(fig, outer[2 * row + 1])
        ax.plot(l)
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.add_subplot(ax)

    if False:
        ax = plt.Subplot(fig, outer[2])
        ax.imshow(dot(a, a.T))
        fig.add_subplot(ax)


def plot_pca_variance(data, k, **kw):
    pca = decomposition.PCA(k)
    pca.fit(data.reshape(len(data), -1))
    plot(pca.explained_variance_, **kw)


def center_rows(images):
    from scipy.linalg import norm

    shape = images.shape
    images = images.reshape(len(images), -1)
    images = images - np.mean(images, axis=1)[:, np.newaxis]
    images /= norm(images, axis=1)[:, np.newaxis]
    images.reshape(*shape)
    return images


def flat_rows(images):
    return images.reshape(len(images), -1)