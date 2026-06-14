def load_gal_for_imshow(gal_id, img_orientation, d):
    """Extract and preprocess one galaxy image for matplotlib imshow.

    Applies min-max scaling to [0, 255], drops the vmap channel, and
    permutes from NCHW to NHWC so imshow interprets the last axis as
    channels.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID (e.g. 470 for object_470).
    img_orientation : str
        Projection key, e.g. 'projection_xy'.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'obs_sorted', 'orientations', and 'X_proc' (the preprocessed
        image tensor of shape (N, C, H, W), populated by the caller
        before invoking any visual_checks function).

    Returns
    -------
    numpy.ndarray of shape (1, H, W, 3), dtype int, or None if the
    requested galaxy and orientation are absent from the dataset.
    """
    import numpy as np
    import torch
    from gallearn import preprocessing

    obj_str = 'object_' + str(gal_id)
    is_obj = d['obs_sorted'] == obj_str
    is_orientation = d['orientations'] == img_orientation
    if np.sum(is_obj & is_orientation) == 0:
        print(f'{obj_str}, {img_orientation} is not in the data')
        return None
    x = d['X_proc'][is_obj & is_orientation]
    x = preprocessing.min_max_scale_255(x)
    # Drop the vmap channel; keep only the 3 RGB channels.
    x = x[:, :3]
    # Dataset is in nchw, c is in rgb (rgu, really). Permute to nhwc so
    # imshow interprets the last axis as channels.
    x = x.permute(0, 2, 3, 1).to(torch.int)
    x = x.cpu().detach().numpy()
    return x


def show_gal_fr_X(gal_id, d):
    """Show all 11 projections of a galaxy in a 3x4 subplot grid.

    Row 0 shows the three standard projections (xy, yz, zx) with x/y/z
    axis labels; the fourth cell in row 0 is hidden. Rows 1 and 2 show
    the eight octant projections with axes hidden and the projection name
    as the subplot title.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'X_proc' (see load_gal_for_imshow).

    Returns
    -------
    None
    """
    import matplotlib.pyplot as plt

    # (projection suffix, xlabel, ylabel) -- empty strings mean hide axes
    standard = [
        ('xy', 'x', 'y'),
        ('yz', 'y', 'z'),
        ('zx', 'z', 'x'),
    ]
    octant_row1 = ['ppp', 'ppm', 'pmp', 'pmm']
    octant_row2 = ['mpp', 'mpm', 'mmp', 'mmm']

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(20., 15.))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    for col, (suffix, xlabel, ylabel) in enumerate(standard):
        ax = axs[0, col]
        x = load_gal_for_imshow(gal_id, f'projection_{suffix}', d)
        ax.set_title(suffix)
        if x is not None:
            ax.imshow(x[0])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )
    axs[0, 3].set_visible(False)

    for row_idx, octant_row in enumerate([octant_row1, octant_row2]):
        for col, suffix in enumerate(octant_row):
            ax = axs[row_idx + 1, col]
            x = load_gal_for_imshow(
                gal_id, f'projection_{suffix}', d
            )
            ax.set_title(suffix)
            if x is not None:
                ax.imshow(x[0])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()


def vmap_vs_image(gal_id, orientation, d):
    """Show the vmap channel alongside the corresponding galaxy image.

    Displays three panels: a pcolormesh of the vmap (seismic_r colormap,
    symmetric vmin/vmax), an imshow of the raw vmap tensor, and an imshow
    of the band_r (channel 0) image in grayscale.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID.
    orientation : str
        Projection key, e.g. 'projection_yz'.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'X_proc' (see load_gal_for_imshow). Channel 3 of X_proc must be
        the vmap channel.

    Returns
    -------
    None
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    obj_str = 'object_' + str(gal_id)
    is_obj = d['obs_sorted'] == obj_str
    is_orientation = d['orientations'] == orientation
    mask = is_obj & is_orientation
    assert mask.sum() == 1
    vmap = d['X_proc'][mask, 3][0]
    img = d['X_proc'][mask, 0][0]
    galname = d['obs_sorted'][mask]
    vmax = np.nanmax(vmap)
    vmin = -1. * vmax

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    ax = axs[0]
    ax.pcolormesh(
        # pcolormesh expects the top left to be at [-1, 0].
        torch.flip(vmap, dims=(0,)),
        cmap=plt.cm.seismic_r,
        vmin=vmin,
        vmax=vmax
    )
    fig.suptitle(galname)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')

    axs[1].imshow(vmap)

    ax = axs[2]
    ax.imshow(
        img,
        interpolation='none',
        cmap='gray',
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_aspect('equal')

    plt.show()


def vmap_img_overlay(
        gal_id,
        img_orientation,
        gas_min_sden,
        star_min_sden,
        d):
    """Overlay a live-computed vmap on a galaxy image to check alignment.

    Computes a velocity map from raw particle data using
    uci_tools.vel_map.plot, then displays three panels: the raw vmap,
    the galaxy image with the vmap blended on top (opacity proportional
    to distance from white), and the galaxy image alone. Axis labels show
    physical coordinates in kpc.

    Parameters
    ----------
    gal_id : int
        Integer galaxy ID.
    img_orientation : str
        Projection key, e.g. 'projection_yz'. Must be one of
        'projection_xy', 'projection_yz', or 'projection_zx'.
    gas_min_sden : float
        Minimum gas surface density threshold for the vmap computation.
    star_min_sden : float
        Minimum stellar surface density threshold for the vmap
        computation.
    d : dict
        Dataset dict returned by preprocessing.load_data. Must contain
        'X_proc' (see load_gal_for_imshow).

    Returns
    -------
    None
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import uci_tools
    from gallearn import config

    if img_orientation == 'projection_xy':
        horiz_axis, vert_axis = (0, 1)
    elif img_orientation == 'projection_yz':
        horiz_axis, vert_axis = (1, 2)
    elif img_orientation == 'projection_zx':
        horiz_axis, vert_axis = (2, 0)
    else:
        raise ValueError('That `img_orientation` is not an option.')

    obj_str = 'object_' + str(gal_id)
    x = load_gal_for_imshow(gal_id, img_orientation, d)

    super_dir = config.config.get('paths', 'firebox_data_dir')
    firebox_snap = config.config.get('paths', 'firebox_snap')
    obj_path = os.path.join(
        super_dir,
        firebox_snap,
        'particles_within_Rvir_' + obj_str + '.hdf5',
    )

    fov = uci_tools.firebox_io.get_fov(gal_id)

    star_pos, star_vs, star_ms, _ = uci_tools.firebox_io.load_particles(
        'stellar',
        obj_path,
        only_bound=False,
    )
    gas_pos, gas_vs, gas_ms, _ = uci_tools.firebox_io.load_particles(
        'gas',
        obj_path,
        only_bound=False,
    )

    # Apply FOV filter in the requested projection plane.
    star_in_fov = (
        np.linalg.norm(star_pos[:, [horiz_axis, vert_axis]], axis=1)
        <= fov / 2.
    )
    star_pos = star_pos[star_in_fov]
    star_vs = star_vs[star_in_fov]
    star_ms = star_ms[star_in_fov]

    gas_in_fov = (
        np.linalg.norm(gas_pos[:, [horiz_axis, vert_axis]], axis=1)
        <= fov / 2.
    )
    gas_pos = gas_pos[gas_in_fov]
    gas_vs = gas_vs[gas_in_fov]
    gas_ms = gas_ms[gas_in_fov]

    (
        colormesh_gas,
        colormesh_stars,
        x_edges_gas,
        z_edges_gas,
        x_edges_stars,
        z_edges_stars,
        pcol_gas,
        pcol_stars,
    ) = uci_tools.vel_map.plot(
        star_pos,
        star_vs,
        star_ms,
        gas_pos,
        gas_vs,
        gas_ms,
        obj_str,
        '600',
        horiz_axis=horiz_axis,
        vert_axis=vert_axis,
        res=256,
        min_gas_cden=gas_min_sden,
        min_stars_cden=star_min_sden,
        show_plot=False
    )

    norm = plt.Normalize(*pcol_gas.get_clim())

    rgba = torch.tensor(
        pcol_gas.get_cmap()(norm(colormesh_gas)) * 255,
        dtype=torch.int
    )
    rgba_unmod = rgba.clone()

    rgb = rgba[..., :3]
    dist_to_white = torch.linalg.norm(rgb - 255., axis=-1)
    dist_to_white = (
        (dist_to_white - dist_to_white.min())
        / (dist_to_white.max() - dist_to_white.min())
    )
    rgba = rgba.to(torch.float)
    rgba[..., 3] *= dist_to_white ** 0.8
    rgba = rgba.to(torch.int)

    fig = plt.figure(figsize=(20, 10))
    axs = fig.subplots(1, 3)

    axs[0].imshow(
        rgba_unmod,
        interpolation='nearest',
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )

    axs[1].imshow(
        x[0],
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )
    axs[1].imshow(
        rgba,
        interpolation='nearest',
        alpha=0.6,
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )

    axs[2].imshow(
        x[0],
        extent=[
            x_edges_gas.min(),
            x_edges_gas.max(),
            z_edges_gas.min(),
            z_edges_gas.max(),
        ]
    )
    axis_labels = ['x', 'y', 'z']
    for ax in axs:
        ax.set_xlabel(
            '{0} [kpc]'.format(axis_labels[horiz_axis]), fontsize=13
        )
        ax.set_ylabel(
            '{0} [kpc]'.format(axis_labels[vert_axis]), fontsize=13
        )

    plt.show()
