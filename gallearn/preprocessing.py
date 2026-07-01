import contextlib
import h5py
import time
import math
import torch


@contextlib.contextmanager
def _open_hdf5(path):
    # h5py.File.close() raises RuntimeError (EBADF) on the second
    # _close_open_objects call when the file lives on an SMB share.
    # That call targets secondary HDF5 file-type objects (e.g. external
    # links). The first call (datasets/groups) and f.id.close() both
    # succeed. We replicate the two-step close manually so the main file
    # descriptor is always released even when the secondary cleanup fails.
    f = h5py.File(path, 'r', locking=False)
    try:
        yield f
    finally:
        if f.id.valid:
            f.id._close_open_objects(h5py.h5f.OBJ_LOCAL | ~h5py.h5f.OBJ_FILE)
            try:
                f.id._close_open_objects(h5py.h5f.OBJ_LOCAL | h5py.h5f.OBJ_FILE)
            except RuntimeError:
                pass
            f.id.close()
            h5py._objects.nonlocal_close()


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
    with _open_hdf5(data_path) as f:
        # Expects (N, C, H, W) layout. Run scripts/transpose_hdf5.py
        # first if the file is still in Julia order (H, W, C, N).
        X = torch.FloatTensor(f['X'][()])
        obs_sorted = np.array(f['obs_sorted'], dtype=str)
        orientations = np.array(f['orientations'], dtype=str)
        file_names = np.array(f['file_names'], dtype=str)
        # Expects (N, 1) layout. Run scripts/transpose_hdf5.py first if the
        # file
        # still has Julia order (1, N).
        ys_sorted = torch.FloatTensor(np.array(f['ys_sorted']))
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


def load_metadata(fname):
    '''
    Load everything except X from the HDF5 file. Returns a tuple
    (d, N, hdf5_path) where d is a dict with keys 'ys_sorted',
    'obs_sorted', 'orientations', 'file_names', N is the number of samples,
    and hdf5_path is the full path to the HDF5 file.
    '''
    import os
    import numpy as np

    from . import config

    hdf5_path = os.path.join(
        config.config['gallearn_paths']['project_data_dir'],
        fname
    )

    with _open_hdf5(hdf5_path) as f:
        N = f['X'].shape[0]
        obs_sorted = np.array(f['obs_sorted'], dtype=str)
        orientations = np.array(f['orientations'], dtype=str)
        file_names = np.array(f['file_names'], dtype=str)
        ys_sorted = torch.FloatTensor(
            np.array(f['ys_sorted'])
        )
    d = {
        'obs_sorted': obs_sorted,
        'orientations': orientations,
        'file_names': file_names,
        'ys_sorted': ys_sorted
    }

    return d, N, hdf5_path


def compute_scaling_stats(hdf5_path, N, stretch=1.e-5, chunk_size=256):
    '''
    Compute per-channel mean and std for transformed X data without loading
    the full dataset into memory.

    Uses a two-pass chunked approach over the HDF5 file. Both passes share
    a single file handle. The transforms applied before computing stats
    match sasinh_imgs_sscale_vmaps: channels 0-2 get asinh(stretch * x),
    and channel 3 (vmap) has NaN replaced with 0.

    Parameters
    ----------
    hdf5_path : str or Path
        Path to the HDF5 file. Must contain an 'X' dataset of shape
        (N_total, 4, H, W).
    N : int
        Number of samples to use. Reads indices [0, N) from the dataset.
    stretch : float, optional
        Asinh stretch factor applied to image channels 0-2. Default 1e-5.
    chunk_size : int, optional
        Number of samples to load per iteration. Default 256.

    Returns
    -------
    means : torch.Tensor, shape (4,)
        Per-channel mean of the transformed data over the N samples.
    stds : torch.Tensor, shape (4,)
        Per-channel standard deviation (population) of the transformed
        data over the N samples.
    '''
    channel_sum = torch.zeros(4, dtype=torch.float64)
    channel_count = torch.zeros(4, dtype=torch.float64)
    channel_sq_sum = torch.zeros(4, dtype=torch.float64)

    # Both passes share one file handle to avoid the h5py EBADF error that
    # occurs when the same file is opened, read in many chunks, closed, and
    # immediately re-opened for a second pass.
    with _open_hdf5(hdf5_path) as f:
        X_dset = f['X']

        # Pass 1: compute means
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # HDF5 shape is (N, C, H, W); read a chunk of samples
            chunk = torch.FloatTensor(X_dset[start:end])

            # Apply transforms
            chunk[:, :3] = torch.asinh(stretch * chunk[:, :3])
            vmap = chunk[:, 3:4]
            vmap[torch.isnan(vmap)] = 0.0
            chunk[:, 3:4] = vmap

            for c in range(4):
                vals = chunk[:, c]
                channel_sum[c] += vals.to(torch.float64).sum()
                channel_count[c] += vals.numel()

        means = (channel_sum / channel_count).float()

        # Pass 2: compute stds
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = torch.FloatTensor(X_dset[start:end])

            chunk[:, :3] = torch.asinh(stretch * chunk[:, :3])
            vmap = chunk[:, 3:4]
            vmap[torch.isnan(vmap)] = 0.0
            chunk[:, 3:4] = vmap

            for c in range(4):
                vals = chunk[:, c].to(torch.float64)
                diff = vals - means[c].to(torch.float64)
                channel_sq_sum[c] += (diff ** 2).sum()

    stds = (channel_sq_sum / channel_count).sqrt().float()

    return means, stds


class LazyGalaxyDataset(torch.utils.data.Dataset):
    '''
    A Dataset that reads galaxy images from an HDF5 file on demand, one sample
    at a time, instead of loading the full dataset into memory.
    '''
    def __init__(self, hdf5_path, indices, means, stds, stretch, ys, rs):
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.means = means
        self.stds = stds
        self.stretch = stretch
        self.ys = ys
        self.rs = rs
        self._file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r', locking=False)
        hdf5_idx = self.indices[idx].item()
        # HDF5 shape is (N, C, H, W); read one sample -> (C, H, W)
        x = torch.FloatTensor(self._file['X'][hdf5_idx])

        # Apply same transforms as sasinh_imgs_sscale_vmaps with precomputed
        # stats.
        x[:3] = torch.asinh(self.stretch * x[:3])
        x[3][torch.isnan(x[3])] = 0.0
        for c in range(4):
            x[c] = (x[c] - self.means[c]) / self.stds[c]

        return x, self.rs[idx], self.ys[idx]

    def __getstate__(self):
        # h5py file objects cannot be pickled. When num_workers > 0, the
        # DataLoader pickles this dataset to send it to each worker process
        # (macOS uses "spawn" rather than "fork"). If anything has called
        # __getitem__ before the DataLoader spawns workers, _file will be
        # an open h5py._hl.files.File object and pickling would fail. We
        # return a copy of the state dict with _file replaced by None so
        # that pickle sees a plain None instead of an unpicklable h5py
        # handle. Each worker then re-opens the file lazily on its first
        # __getitem__ call.
        state = self.__dict__.copy()
        state['_file'] = None
        return state

    def __del__(self):
        if self._file is not None:
            self._file.close()


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
    '''
    Apply asinh to the dataset with the given stretch and standardize the
    result along axis 1 e.g. the color channels of a
    NCHW (number-channel-height-width) dataset

    Parameters
    ----------
    X: torch.tensor, shape (N_obs, N_chan, h, w)
        Dataset to process.
    stretch: float
        The value by which to multiply X before applying asinh.
    return_distrib: bool, default False
        If True, return the means and standard deviations of each channel so
        the user can apply them later without recalculating
    means: torch.tensor of floats, shape (N_chan,), default None
        The means to apply to each channel when standardizing
    stds: torch.tensor of floats, shape (N_chan,), default None
        The standard deviations to appy to each channel when standardizing

    Returns
    -------
    X: torch.tensor, shape (N_obs, N_chan, h, w)
        The standardized dataset.
    means: torch.tensor, shape (N_chan,)
        The calculated mean of each channel. Only returned when
        return_distrib=True.
    stds: torch.tensor, shape (N_chan,)
        The calculated standard deviation of each channel. Only returned
        when return_distrib=True.
    '''
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
    once, on a per-channel (r, g, u) basis. Given that the X input spans ~8
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
    '''
    Standardize a dataset along axis 1 e.g. the color channels of a NCHW 
    (number-channel-height-width) dataset

    Parameters
    ----------
    X: torch.tensor, shape (N_obs, N_chan, h, w)
        Dataset to standardize
    return_distrib: bool, default False
        If True, return the means and standard deviations of each channel so
        the user can apply them later without recalculating
    means: torch.tensor of floats, shape (N_chan,), default None
        The means to apply to each channel when standardizing
    stds: torch.tensor of floats, shape (N_chan,), default None
        The standard deviations to appy to each channel when standardizing

    Returns
    -------
    X: torch.tensor, shape (N_obs, N_chan, h, w)
        The standardized dataset.
    means: torch.tensor, shape (N_chan,)
        The calculated mean of each channel. Only returned when
        return_distrib=True.
    stds: torch.tensor, shape (N_chan,)
        The calculated standard deviation of each channel. Only returned
        when return_distrib=True.
    '''
    import torch
    X = X.detach().clone()
    if means is None:
        means = torch.zeros(X.shape[1])
        calc_means = True
    else:
        calc_means = False
    if stds is None:
        stds = torch.zeros(X.shape[1])
        calc_stds = True
    else:
        calc_stds = False
    for i in range(X.shape[1]):
        if calc_stds:
            std = X[:, i].std()
        else:
            std = stds[i]
        if calc_means:
            mean = X[:, i].mean()
        else:
            mean = means[i]
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
        colors = ['C3', 'C2', 'C0'],
        labels = ['r', 'g', 'u']
    else:
        colors = ['C3']
        labels = ['r']
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
        colors = ['C3', 'C2', 'C0'],
        labels = ['r', 'g', 'u']
    else:
        colors = ['C3']
        labels = ['r']
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
    from . import config
    rcParams['axes.titlesize'] = 8.

    d = load_data(config.config['gallearn_paths']['dataset'])
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
    import numpy as np
    from . import config

    d = load_data(config.config['gallearn_paths']['dataset'])
    ssfrs = d['ys_sorted']
    #iszero = ssfrs == 0.
    #print(len(ssfrs))
    #print(iszero.sum())
    #new_ys = torch.asinh(ssfrs * 1.e12)
    new_ys, means, stds = std_asinh(ssfrs, 1.e11, return_distrib=True)
    print(means, stds)
    new_ys = new_ys.flatten()
    print(new_ys.min(), new_ys.max(), torch.median(new_ys))
    isnan = torch.isnan(ssfrs)
    print(isnan.sum())
    #print(d['obs_sorted'][isnan])
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(new_ys)
    ax.set_xlabel('$\dfrac{SFR}{M_\star}\;[\mathrm{Gyr}^{-1}]$')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

    return None


if __name__ == '__main__':
    test(save=True)
