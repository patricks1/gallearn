import h5py
import time
import math

def load_data():
    import torch
    import numpy as np

    data_path = "/DFS-L/DATA/cosmo/pstaudt/gallearn/gallearn_data.h5"

    start = time.time()
    with h5py.File(data_path, 'r') as f:
        X = torch.FloatTensor(np.array(f['X'])).permute(3, 2, 1, 0)
        obs_sorted = list(f['obs_sorted'])
        file_names = list(f['file_names'])
        ys_sorted = torch.FloatTensor(np.array(f['ys_sorted']))
    d = {
        'X': X,
        'obs_sorted': obs_sorted,
        'file_names': file_names,
        'ys_sorted': ys_sorted
    }
    end = time.time()
    elapsed = end - start
    minutes = math.floor(elapsed / 60.)
    print('{0:0.0f} min, {1:0.1f} s to load'.format(
        minutes, 
        elapsed - minutes * 60.
    ))

    return d

def min_max_scale(X):
    X = X.detach().clone()
    for i in range(X.shape[1]):
        X[:, i] = 2. * (
            (X[:, i] - X[:, i].min()) 
            / (X[:, i].max() - X[:, i].min()) 
            - 0.5
        )
    return X

def std_scale(X):
    X = X.detach().clone()
    for i in range(X.shape[1]):
        std = X[:, i].std()
        mean = X[:, i].mean()
        X[:, i] -= mean
        X[:, i] /= std
    return X 

def plt_distrib_of_means(ax, X, title=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    means = X.flatten(2, 3).mean(2).transpose(1, 0)

    minimum = means.min()
    maximum = means.max()
    bins = np.linspace(minimum, maximum, 21)
    heights = torch.zeros(3, 20)
    edges = torch.zeros(3, 21)
    for i, m in enumerate(means):
        heights[i], edges[i] = torch.histogram(m, bins=bins)
    for h, e, c, l in zip(
                heights, 
                edges,
                ['C0', 'C2', 'C3'],
                ['U', 'G', 'R']
            ):
        ax.stairs(
            h,
            e,
            color=c,
            label=l
        )
    ax.set_yscale('log')
    ax.set_title(title)

    print('Added {0:s} to figure.'.format(title.lower()))
    return None

def plt_distrib(ax, X, title=None):
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    Xpermflat = X.permute(1, 0, 2, 3).flatten(1,3)
    heights = torch.zeros(3, 20)
    edges = torch.zeros(3, 21)
    min_ = Xpermflat.min()
    max_ = Xpermflat.max()
    bins = np.linspace(min_, max_, 21) 
    for i in range(Xpermflat.shape[0]):
        hist = torch.histogram(Xpermflat[i], bins=bins)
        heights[i], edges[i] = hist

    for h, e, c, l in zip(
                heights, 
                edges,
                ['C0', 'C2', 'C3'],
                ['U', 'G', 'R']
            ):
        ax.stairs(
            h,
            e,
            color=c,
            label=l
        )
    ax.set_yscale('log')
    ax.set_title(title)

    print('Added {0:s} to figure.'.format(title.lower()))
    return None

def test(save=False):
    import torch
    from matplotlib import pyplot as plt

    X = load_data()['X']
    Xstd = std_scale(X)
    Xminmax = min_max_scale(X)

    logX = torch.log10(X)
    iszero = X == 0.
    logX[iszero] = logX[~iszero].min()
    
    Xlogstd = std_scale(logX)
    Xlogminmax = min_max_scale(logX)

    hspace = 0.4

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5), dpi=140, sharey=True)
    fig.subplots_adjust(hspace=hspace, wspace=0.)
    axs = axs.ravel()
    plt_distrib_of_means(axs[0], X, 'Not scaled')
    plt_distrib_of_means(axs[1], Xstd, 'Standardized')
    plt_distrib_of_means(axs[2], Xminmax, 'Min-max scaled')
    plt_distrib_of_means(axs[3], logX, 'Logged')
    plt_distrib_of_means(axs[4], Xlogstd, 'Logged and standardized')
    plt_distrib_of_means(axs[5], Xlogminmax, 'Logged and min-max scaled')
    axs[0].legend()
    for i in [0, 3]:
        axs[i].set_ylabel('Num images')
    axs[-2].set_xlabel('Mean image value')
    if save:
        plt.savefig('mean_distribs.png', dpi=200)
    plt.show()

    fig, axs = plt.subplots(2, 3, figsize=(10, 5.5), dpi=140, sharey=True)
    fig.subplots_adjust(hspace=hspace, wspace=0.)
    axs=axs.ravel()
    plt_distrib(axs[0], X, 'Not scaled')
    plt_distrib(axs[1], Xstd, 'Standardized')
    plt_distrib(axs[2], Xminmax, 'Min-max scaled')
    plt_distrib(axs[3], logX, 'Logged')
    plt_distrib(axs[4], Xlogstd, 'Logged and standardized')
    plt_distrib(axs[5], Xlogminmax, 'Logged and min-max scaled')
    axs[0].legend()
    for i in [0, 3]:
        axs[i].set_ylabel('Num pixels')
    axs[-2].set_xlabel('Pixel value')
    if save:
        plt.savefig('pixel_distribs.png', dpi=200)
    plt.show()

    return None

if __name__ == '__main__':
    test(save=True)
