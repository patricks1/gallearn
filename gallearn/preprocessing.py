import h5py
import time
import math

def load_data(fname):
    import torch
    import os
    import numpy as np

    from . import config

    data_path = os.path.join(
        config.config['gallearn_paths']['project_data_dir'],
        fname
    )

    start = time.time()
    with h5py.File(data_path, 'r') as f:
        X = f['X'][()]
        X = torch.FloatTensor(X)
        # I saved the data with Julia, which is transposed to the way Python
        # expects, so we must permute.
        X = X.permute(3, 2, 0, 1)
        obs_sorted = np.array(f['obs_sorted'], dtype=str)
        orientations = np.array(f['orientations'], dtype=str)
        file_names = np.array(f['file_names'], dtype=str)
        ys_sorted = torch.FloatTensor(np.array(f['ys_sorted']))
        ys_sorted = torch.FloatTensor(np.array(f['ys_sorted'])).transpose(1, 0)
    d = {
        'X': X,
        'obs_sorted': obs_sorted,
        'orientations': orientations,
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

def sasinh_imgs_sscale_vmaps(X, stretch=1.e-5):
    import torch
    X = X.detach().clone()
    
    X_imgs = X[:, :3]
    X_imgs = std_asinh(X_imgs, stretch)

    X_vmaps = X[:, 3:4]
    isnan = torch.isnan(X_vmaps)
    X_vmaps[isnan] = 0.
    X_vmaps = std_scale(X_vmaps)

    X = torch.cat((X_imgs, X_vmaps), dim=1)

    return X

def std_asinh(X, stretch=1.e-5, return_distrib=False, means=None, stds=None):
    import torch
    X = X.detach().clone()
    X = torch.asinh(stretch * X)
    return std_scale(X, return_distrib, means, stds)

def min_max_scale(X):
    '''
    Min-max scale the data from -1 to 1.
    '''
    X = X.detach().clone()
    for i in range(X.shape[1]):
        X[:, i] = 2. * (
            (X[:, i] - X[:, i].min()) 
            / (X[:, i].max() - X[:, i].min()) 
            - 0.5
        )
    return X

def min_max_scale_255(X):
    '''
    Min-max scale the data from 0 to 255.
    '''
    X = X.detach().clone()
    for i in range(X.shape[1]):
        X[:, i] = 255. * (
            (X[:, i] - X[:, i].min()) 
            / (X[:, i].max() - X[:, i].min()) 
        )
    return X

def new_min_max_scale(X):
    '''
    Min-max scale the data from 0 to 255. Scaling is done for all galaxies at
    once, on a per-channel (u, g, r) basis. Given that the X input spans ~8
    orders of magnitues, only the brightest regions contribute significant
    information. Additionally, any galaxies whose maximum brightness falls many
    orders of magnitude below the brightest galaxy may not show significant
    information post-scaling.
    '''
    X = X.detach().clone()
    for i in range(X.shape[1]):
        X[:, i] = 255. * (
            (X[:, i] - X[:, i].min()) 
            / (X[:, i].max() - X[:, i].min()) 
        )
    return X

def std_scale(X, return_distrib=False, means=None, stds=None):
    import torch
    X = X.detach().clone()
    if means is None:
        means = torch.zeros(X.shape[1])
    if stds is None:
        stds = torch.zeros(X.shape[1])
    for i in range(X.shape[1]):
        std = X[:, i].std()
        mean = X[:, i].mean()
        X[:, i] -= mean
        X[:, i] /= std
        means[i] = mean
        stds[i] = std
    if return_distrib:
        return X, means, stds
    else:
        return X 

def log_min_max_scale(X):
    import torch
    logX = torch.log10(X)
    iszero = X == 0.
    logX[iszero] = logX[~iszero].min()
    Xlogminmax = min_max_scale(logX)
    return Xlogminmax

def plt_distrib_of_means(ax, X, title=None, all_bands=False):
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

    if all_bands:
        colors = ['C2', 'C0', 'C3'],
        labels = ['g', 'u', 'r']
    else:
        colors = ['C2']
        labels = ['g']
    for h, e, c, l in zip(
                heights.detach().cpu().numpy(), 
                edges.detach().cpu().numpy(),
                colors,
                labels
            ):
        ax.stairs(
            h,
            e,
            color=c,
            label=l
        )
    #ax.set_yscale('log')
    ax.set_title(title)
    ax.ticklabel_format(
        style='sci',
        scilimits=(-1,3),
        axis='x',
        useMathText=True
    )
    ax.tick_params(axis='x', labelrotation=45, labelright=False)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')

    if title is not None:
        print('Added {0:s} to figure.'.format(title.lower()))
    return None

def plt_distrib(ax, X, title=None, all_bands=False):
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    Xpermflat = X.permute(1, 0, 2, 3).flatten(1,3)
    heights = torch.zeros(3, 20)
    edges = torch.zeros(3, 21)
    min_ = Xpermflat.min()
    max_ = Xpermflat.max()
    print(min_, max_)
    bins = torch.linspace(
        min_,
        max_,
        21
    )
    for i in range(Xpermflat.shape[0]):
        hist = torch.histogram(Xpermflat[i], bins=bins)
        heights[i], edges[i] = hist

    if all_bands:
        colors = ['C2', 'C0', 'C3'],
        labels = ['g', 'u', 'r']
    else:
        colors = ['C2']
        labels = ['g']
    for h, e, c, l in zip(
                heights.detach().cpu().numpy(), 
                edges.detach().cpu().numpy(),
                colors,
                labels
            ):
        ax.stairs(
            h,
            e,
            color=c,
            label=l
        )
    ax.set_title(title)
    ax.ticklabel_format(
        style='sci',
        scilimits=(-1,3),
        axis='x',
        useMathText=True
    )
    ax.ticklabel_format(
        style='sci',
        scilimits=(-1,3),
        axis='x',
        useMathText=True
    )

    if title is not None:
        print('Added {0:s} to figure.'.format(title.lower()))
    return None

def test(save=False):
    import torch
    from matplotlib import pyplot as plt
    from matplotlib import rcParams
    rcParams['axes.titlesize'] = 8.

    d = load_data('gallearn_data_256x256_3proj_2d_tgt.h5')
    X = d['X']
    Xstd = std_scale(X)
    Xminmax = min_max_scale(X)
    Xminmax256 = new_min_max_scale(X)
    Xasinh = std_asinh(X)

    logX = torch.log10(X)
    iszero = X == 0.
    logX[iszero] = logX[~iszero].min()
    
    Xlogstd = std_scale(logX)
    Xlogminmax = min_max_scale(logX)

    hspace = 0.4

    fig, axs = plt.subplots(2, 4, figsize=(10, 5.5), dpi=140, sharey=True)
    fig.subplots_adjust(hspace=hspace, wspace=0.)
    axs = axs.ravel()
    plt_distrib_of_means(axs[0], X, 'Not scaled')
    plt_distrib_of_means(axs[1], Xstd, 'Standardized')
    plt_distrib_of_means(axs[2], Xminmax, 'Min-max scaled')
    plt_distrib_of_means(axs[3], Xminmax256, 'Min-max scaled (0-255)')
    plt_distrib_of_means(axs[4], logX, 'Logged')
    plt_distrib_of_means(axs[5], Xlogstd, 'Logged and standardized')
    plt_distrib_of_means(axs[6], Xlogminmax, 'Logged and min-max scaled')
    plt_distrib_of_means(axs[7], Xasinh, 'Standardized asinh')
    axs[0].legend()
    for i in [0, 4]:
        axs[i].set_ylabel('Num images')
    axs[-2].set_xlabel('Mean image value')
    #axs[-1].set_axis_off()
    for ax in axs:
        ax.set_yscale('log')
    if save:
        plt.savefig('mean_distribs_log.png', dpi=200)
    for ax in axs:
        ax.set_yscale('linear')
    if save:
        plt.savefig('mean_distribs.png', dpi=200)
    plt.close()

    fig, axs = plt.subplots(2, 4, figsize=(10, 5.5), dpi=140, sharey=True)
    fig.subplots_adjust(hspace=hspace, wspace=0.)
    axs=axs.ravel()
    plt_distrib(axs[0], X, 'Not scaled')
    plt_distrib(axs[1], Xstd, 'Standardized')
    plt_distrib(axs[2], Xminmax, 'Min-max scaled')
    plt_distrib(axs[3], Xminmax256, 'Min-max scaled (0-255)')
    plt_distrib(axs[4], logX, 'Logged')
    plt_distrib(axs[5], Xlogstd, 'Logged and standardized')
    plt_distrib(axs[6], Xlogminmax, 'Logged and min-max scaled')
    plt_distrib(axs[7], Xasinh, 'Standardized asinh')
    axs[0].legend()
    for i in [0, 4]:
        axs[i].set_ylabel('Num pixels')
    axs[-2].set_xlabel('Pixel value')
    #axs[-1].set_axis_off()
    for ax in axs:
        ax.set_yscale('log')
    if save:
        plt.savefig('pixel_distribs_log.png', dpi=200)
    for ax in axs:
        ax.set_yscale('linear')
    if save:
        plt.savefig('pixel_distribs.png', dpi=200)
    plt.close()

    return None

def plt_ssfr():
    import os
    import torch
    import matplotlib.pyplot as plt

    d = load_data('gallearn_data_128x128_3proj_wsat_sfr_tgt.h5')
    ssfrs = d['ys_sorted']#.flatten()
    #iszero = ssfrs == 0.
    #print(len(ssfrs))
    #print(iszero.sum())
    #new_ys = torch.asinh(ssfrs * 1.e12)
    #new_ys = std_asinh(ssfrs, 1.e11).flatten()
    new_ys = ssfrs.flatten()
    print(new_ys.min(), new_ys.max(), torch.median(new_ys))
    #isnan = torch.isnan(ssfrs)
    #print(isnan.sum())
    #print(d['obs_sorted'][isnan])
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(new_ys)
    ax.set_xlabel('$\dfrac{SFR}{M_\star}\;[10^{-8}\,\mathrm{yr}^{-1}]$')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

    return None

if __name__ == '__main__':
    test(save=True)
